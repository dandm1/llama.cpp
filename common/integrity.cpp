#include "integrity.h"

#include "ggml-backend.h"
#include "log.h"

#include "../src/llama-ext.h" // staging API: llama_model_for_each_tensor

#include "ggml-alloc.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <thread>

common_integrity::~common_integrity() {
    devices_free();
}

uint64_t common_integrity::digest_of(const entry & e, int n_threads) {
    if (!e.host || n_threads <= 1 || e.bytes < (size_t) 64 * 1024 * 1024) {
        return ggml_backend_tensor_hash(e.tensor);
    }
    // host memory, in parallel pieces: the digest is order-independent, so the pieces combine with xor
    const size_t n   = e.bytes;
    const size_t cut = ((n / n_threads) / 8) * 8;
    std::vector<uint64_t> parts(n_threads, 0);
    std::vector<std::thread> workers;
    for (int t = 0; t < n_threads; t++) {
        const size_t pos = (size_t) t * cut;
        const size_t len = t + 1 == n_threads ? n - pos : cut;
        workers.emplace_back([&, t, pos, len]() {
            parts[t] = ggml_backend_hash_bytes((const uint8_t *) e.tensor->data + pos, len, pos);
        });
    }
    uint64_t acc = 0;
    for (int t = 0; t < n_threads; t++) {
        workers[t].join();
        acc ^= parts[t];
    }
    return ggml_backend_hash_final(acc, n);
}

double common_integrity::init(const llama_model * model, int n_threads) {
    const int64_t t0 = ggml_time_us();
    entries.clear();
    bytes_device = bytes_host = 0;
    host_cursor = 0;

    llama_model_for_each_tensor(model, [](const ggml_tensor * t, void * ud) -> bool {
        auto * self = (common_integrity *) ud;
        if (t->data == nullptr || t->buffer == nullptr) {
            return true; // not loaded (e.g. the MTP layers without drafting)
        }
        ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
        entry e;
        e.tensor = t;
        e.device = ggml_backend_buffer_name(buf);
        e.host   = ggml_backend_buffer_is_host(buf);
        e.bytes  = ggml_nbytes(t);
        e.digest = 0;
        self->entries.push_back(e);
        return true;
    }, this);

    devices_init();

    bool kernel_checked = false;
    for (auto & e : entries) {
        e.digest = digest_of(e, n_threads);
        (e.host ? bytes_host : bytes_device) += e.bytes;
        if (!e.host && !kernel_checked) {
            // the device kernel against the host definition, once: the two must agree on the same bytes
            kernel_checked = true;
            std::vector<uint8_t> tmp(e.bytes);
            ggml_backend_tensor_get(e.tensor, tmp.data(), 0, e.bytes);
            const uint64_t host_digest = ggml_backend_hash_final(ggml_backend_hash_bytes(tmp.data(), e.bytes, 0), e.bytes);
            if (host_digest != e.digest) {
                LOG_ERR("integrity: the device hash of %s on %s (%016llx) disagrees with the host hash of the same bytes (%016llx); the check cannot be trusted on this backend\n",
                    ggml_get_name(e.tensor), e.device.c_str(), (unsigned long long) e.digest, (unsigned long long) host_digest);
            } else {
                LOG_INF("integrity: device hash of %s on %s agrees with the host hash\n", ggml_get_name(e.tensor), e.device.c_str());
            }
        }
    }
    return (ggml_time_us() - t0) * 1e-3;
}

common_integrity::report common_integrity::verify(int n_threads) {
    const int64_t t0 = ggml_time_us();
    report r;

    // host tensors: the next slice of the rotation
    std::vector<bool> check(entries.size(), false);
    size_t n_host = 0;
    for (size_t i = 0; i < entries.size(); i++) {
        if (!entries[i].host) {
            check[i] = true;
        } else {
            n_host++;
        }
    }
    if (n_host > 0) {
        size_t budget = host_bytes_per_pass == 0 ? SIZE_MAX : host_bytes_per_pass;
        size_t seen = 0;
        while (seen < n_host && budget > 0) {
            host_cursor %= entries.size();
            if (entries[host_cursor].host) {
                check[host_cursor] = true;
                budget = entries[host_cursor].bytes >= budget ? 0 : budget - entries[host_cursor].bytes;
                seen++;
            }
            host_cursor++;
        }
    }

    for (size_t i = 0; i < entries.size(); i++) {
        if (!check[i]) {
            continue;
        }
        const entry & e = entries[i];
        const uint64_t d = digest_of(e, n_threads);
        r.n_checked++;
        r.bytes_checked += e.bytes;
        if (d != e.digest) {
            r.n_mismatch++;
            r.mismatches.push_back(std::string(ggml_get_name(e.tensor)) + " on " + e.device);
            LOG_ERR("integrity: tensor %s on %s changed since load (digest %016llx, was %016llx)\n",
                ggml_get_name(e.tensor), e.device.c_str(), (unsigned long long) d, (unsigned long long) e.digest);
        }
    }
    devices_verify(r);

    r.t_ms = (ggml_time_us() - t0) * 1e-3;
    return r;
}

// ---- the per-device workloads

static void fill_pattern(std::vector<float> & v, uint32_t seed) {
    uint32_t x = seed;
    for (auto & f : v) {
        x = x * 1664525u + 1013904223u;
        f = (float) (x >> 8) / (float) (1u << 24) - 0.5f;
    }
}

void common_integrity::devices_init() {
    devices_free();
    // every device holding a tensor, and the CPU, so the copy check covers host <-> device as well
    std::vector<ggml_backend_dev_t> devs;
    for (const auto & e : entries) {
        ggml_backend_buffer_t buf = e.tensor->view_src ? e.tensor->view_src->buffer : e.tensor->buffer;
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buf));
        if (dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU && std::find(devs.begin(), devs.end(), dev) == devs.end()) {
            devs.push_back(dev);
        }
    }
    if (ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
        devs.push_back(cpu);
    }

    const int64_t n_mat = 256;        // 256x256 f32 matmul: a few MFLOP, every SM gets a tile
    const int64_t n_pat = 4*1024*1024; // 16 MiB pattern, enough to cross a link in one DMA

    std::vector<float> a((size_t) n_mat*n_mat), b((size_t) n_mat*n_mat), pat((size_t) n_pat);
    fill_pattern(a, 1); fill_pattern(b, 2); fill_pattern(pat, 3);
    pattern_digest = ggml_backend_hash_final(ggml_backend_hash_bytes(pat.data(), pat.size()*sizeof(float), 0), pat.size()*sizeof(float));

    for (ggml_backend_dev_t dev : devs) {
        device_check d;
        d.dev  = dev;
        d.name = ggml_backend_dev_name(dev);
        d.backend = ggml_backend_dev_init(dev, nullptr);
        if (!d.backend) {
            LOG_WRN("integrity: no backend for %s, its compute and copy checks are skipped\n", d.name.c_str());
            continue;
        }
        ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
        d.ctx = ggml_init(ip);
        ggml_tensor * ta = ggml_new_tensor_2d(d.ctx, GGML_TYPE_F32, n_mat, n_mat);
        ggml_tensor * tb = ggml_new_tensor_2d(d.ctx, GGML_TYPE_F32, n_mat, n_mat);
        d.out     = ggml_mul_mat(d.ctx, ta, tb);
        d.pattern = ggml_new_tensor_1d(d.ctx, GGML_TYPE_F32, n_pat);
        d.landing = ggml_new_tensor_1d(d.ctx, GGML_TYPE_F32, n_pat);
        ggml_set_name(ta, "integrity_a"); ggml_set_name(tb, "integrity_b"); ggml_set_name(d.out, "integrity_out");
        ggml_set_name(d.pattern, "integrity_pattern"); ggml_set_name(d.landing, "integrity_landing");
        d.buf = ggml_backend_alloc_ctx_tensors(d.ctx, d.backend);
        if (!d.buf) {
            LOG_WRN("integrity: could not allocate the check tensors on %s, its compute and copy checks are skipped\n", d.name.c_str());
            ggml_free(d.ctx); ggml_backend_free(d.backend);
            continue;
        }
        ggml_backend_tensor_set(ta, a.data(), 0, ggml_nbytes(ta));
        ggml_backend_tensor_set(tb, b.data(), 0, ggml_nbytes(tb));
        ggml_backend_tensor_set(d.pattern, pat.data(), 0, ggml_nbytes(d.pattern));
        d.gf = ggml_new_graph(d.ctx);
        ggml_build_forward_expand(d.gf, d.out);
        if (ggml_backend_graph_compute(d.backend, d.gf) != GGML_STATUS_SUCCESS) {
            LOG_WRN("integrity: the check matmul does not run on %s, its compute check is skipped\n", d.name.c_str());
            d.out = nullptr;
        } else {
            d.out_digest = ggml_backend_tensor_hash(d.out);
        }
        devices.push_back(d);
    }
    LOG_INF("integrity: compute and copy checks on %zu devices, %zu copy paths\n", devices.size(), devices.size()*(devices.size() - 1));
}

void common_integrity::devices_free() {
    for (auto & d : devices) {
        if (d.buf)     ggml_backend_buffer_free(d.buf);
        if (d.ctx)     ggml_free(d.ctx);
        if (d.backend) ggml_backend_free(d.backend);
    }
    devices.clear();
}

void common_integrity::devices_verify(report & r) {
    for (auto & d : devices) {
        if (d.out) {
            if (ggml_backend_graph_compute(d.backend, d.gf) != GGML_STATUS_SUCCESS) {
                r.n_compute_fail++;
                LOG_ERR("integrity: the check matmul failed to run on %s\n", d.name.c_str());
            } else {
                const uint64_t h = ggml_backend_tensor_hash(d.out);
                if (h != d.out_digest) {
                    r.n_compute_fail++;
                    LOG_ERR("integrity: the check matmul on %s gives a different result than at load (digest %016llx, was %016llx)\n",
                        d.name.c_str(), (unsigned long long) h, (unsigned long long) d.out_digest);
                }
            }
        }
    }
    for (auto & src : devices) {
        for (auto & dst : devices) {
            if (&src == &dst) {
                continue;
            }
            ggml_backend_tensor_memset(dst.landing, 0, 0, ggml_nbytes(dst.landing));
            ggml_backend_tensor_copy(src.pattern, dst.landing);
            r.n_copy++;
            const uint64_t h = ggml_backend_tensor_hash(dst.landing);
            if (h != pattern_digest) {
                r.n_copy_fail++;
                LOG_ERR("integrity: a %zu MiB copy from %s to %s arrived changed (digest %016llx, expected %016llx)\n",
                    ggml_nbytes(src.pattern) >> 20, src.name.c_str(), dst.name.c_str(), (unsigned long long) h, (unsigned long long) pattern_digest);
            }
        }
    }
}

// ---- telemetry

static std::string bus_tail(const std::string & id) {
    // cuda reports 0000:21:00.0, nvidia-smi 00000000:21:00.0: compare from the bus number on
    const size_t c = id.find(':');
    std::string t = c == std::string::npos ? id : id.substr(c + 1);
    for (auto & ch : t) ch = (char) tolower(ch);
    return t;
}

std::vector<common_integrity::telemetry_sample> common_integrity::telemetry() {
    std::vector<telemetry_sample> out;
    if (!telemetry_available) {
        return out;
    }
    auto run = [](const char * cmd) -> std::string {
        std::string s;
        FILE * f = popen(cmd, "r");
        if (!f) return s;
        char line[1024];
        while (fgets(line, sizeof(line), f)) s += line;
        pclose(f);
        return s;
    };
    const std::string q = run("nvidia-smi --query-gpu=pci.bus_id,temperature.gpu,temperature.memory,clocks.sm,clocks.mem,power.draw,"
                              "clocks_event_reasons.active,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader,nounits 2>/dev/null");
    if (q.empty()) {
        telemetry_available = false; // no nvidia-smi, or not an NVIDIA box: say so once
        LOG_INF("integrity: nvidia-smi not available, no telemetry line\n");
        return out;
    }
    // replays per bus id, from the full report
    std::map<std::string, long long> replays;
    {
        const std::string full = run("nvidia-smi -q 2>/dev/null");
        std::istringstream in(full);
        std::string line, cur;
        while (std::getline(in, line)) {
            if (line.rfind("GPU ", 0) == 0) {
                cur = bus_tail(line.substr(4));
            } else if (line.find("Replays Since Reset") != std::string::npos) {
                const size_t c = line.find(':');
                if (c != std::string::npos && !cur.empty()) replays[cur] = atoll(line.c_str() + c + 1);
            }
        }
    }
    // device names by bus id
    std::map<std::string, std::string> names;
    for (const auto & d : devices) {
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(d.dev, &props);
        if (props.device_id) names[bus_tail(props.device_id)] = d.name;
    }
    std::istringstream in(q);
    std::string line;
    while (std::getline(in, line)) {
        std::vector<std::string> f;
        std::istringstream ls(line);
        std::string cell;
        while (std::getline(ls, cell, ',')) {
            cell.erase(0, cell.find_first_not_of(" \t"));
            cell.erase(cell.find_last_not_of(" \t\r") + 1);
            f.push_back(cell);
        }
        if (f.size() < 9) continue;
        telemetry_sample s;
        s.bus_id   = bus_tail(f[0]);
        s.device   = names.count(s.bus_id) ? names[s.bus_id] : "";
        if (s.device.empty()) continue; // a card the model does not use
        s.temp_gpu = atoi(f[1].c_str());
        s.temp_mem = f[2] == "N/A" ? -1 : atoi(f[2].c_str());
        s.clk_sm   = atoi(f[3].c_str());
        s.clk_mem  = atoi(f[4].c_str());
        s.power_w  = atof(f[5].c_str());
        s.throttle = strtoull(f[6].c_str(), nullptr, 16);
        s.link_gen = atoi(f[7].c_str());
        s.link_width = atoi(f[8].c_str());
        if (replays.count(s.bus_id)) {
            s.replays = replays[s.bus_id];
            if (replays_prev.count(s.bus_id)) s.replays_delta = s.replays - replays_prev[s.bus_id];
            replays_prev[s.bus_id] = s.replays;
        }
        out.push_back(s);
    }
    return out;
}

bool common_integrity::inject(const std::string & name, std::string & error) {
    for (const auto & e : entries) {
        if (name != ggml_get_name(e.tensor)) {
            continue;
        }
        if (e.device.find("Mapped") != std::string::npos) {
            error = "tensor " + name + " is memory-mapped read-only (" + e.device + "); pick a device tensor or load with --load-mode none";
            return false;
        }
        const size_t off = e.bytes / 2;
        uint8_t b = 0;
        ggml_backend_tensor_get(e.tensor, &b, off, 1);
        b ^= 0x01;
        ggml_backend_tensor_set(const_cast<ggml_tensor *>(e.tensor), &b, off, 1);
        LOG_WRN("integrity: flipped one bit of %s on %s at byte %zu (--integrity-inject), the next check must report it\n",
            name.c_str(), e.device.c_str(), off);
        return true;
    }
    error = "no tensor named " + name + "; device tensors include";
    int shown = 0;
    for (const auto & e : entries) {
        if (!e.host && e.bytes >= (size_t) 1024 * 1024 && shown < 4) {
            error += std::string(shown ? ", " : " ") + ggml_get_name(e.tensor);
            shown++;
        }
    }
    return false;
}
