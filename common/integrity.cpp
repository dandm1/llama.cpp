#include "integrity.h"

#include "ggml-backend.h"
#include "log.h"

#include "../src/llama-ext.h" // staging API: llama_model_for_each_tensor

#include <algorithm>
#include <thread>

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
    r.t_ms = (ggml_time_us() - t0) * 1e-3;
    return r;
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
