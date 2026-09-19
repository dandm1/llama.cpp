#include "measure.h"

#include "build-info.h"
#include "common.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>

using json = nlohmann::ordered_json;

//
// fingerprint
//

std::string fit_advisor_device_fingerprint::key() const {
    std::ostringstream ss;
    ss << backend << "|" << name << "|" << description << "|" << device_id << "|" << total_memory << "|" << build_commit;
    if (n_threads > 0) {
        ss << "|t" << n_threads;
    }
    return ss.str();
}

fit_advisor_device_fingerprint fit_advisor_fingerprint(ggml_backend_dev_t dev, int n_threads) {
    fit_advisor_device_fingerprint fp;
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);

    fp.name         = props.name ? props.name : "";
    fp.description  = props.description ? props.description : "";
    fp.device_id    = props.device_id ? props.device_id : "";
    fp.backend      = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
    fp.total_memory = (int64_t) props.memory_total;
    fp.build_commit = llama_commit();
    fp.n_threads    = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU ? n_threads : 0;
    return fp;
}

//
// benchmark harness: build a small graph, allocate it on the backend, fill the inputs, time it
//

namespace {

struct bench_result {
    bool   supported = false;
    double us_per_run = 0;
    int    n_runs = 0;
    std::string reason;
};

// fill a leaf tensor with random data of its type; quantized types go through the reference quantizer
void fill_random(ggml_tensor * t, std::mt19937 & rng) {
    const int64_t nels = ggml_nelements(t);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(nels);
    for (auto & x : data) {
        x = dist(rng);
    }

    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, data.data(), 0, nels * sizeof(float));
        return;
    }
    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> buf(nels);
        ggml_fp32_to_fp16_row(data.data(), buf.data(), nels);
        ggml_backend_tensor_set(t, buf.data(), 0, nels * sizeof(ggml_fp16_t));
        return;
    }
    if (t->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> buf(nels);
        ggml_fp32_to_bf16_row(data.data(), buf.data(), nels);
        ggml_backend_tensor_set(t, buf.data(), 0, nels * sizeof(ggml_bf16_t));
        return;
    }
    if (ggml_is_quantized(t->type)) {
        const int64_t n_per_row = t->ne[0];
        const int64_t nrows     = nels / n_per_row;
        std::vector<uint8_t> buf(ggml_nbytes(t));
        ggml_quantize_chunk(t->type, data.data(), buf.data(), 0, nrows, n_per_row, nullptr);
        ggml_backend_tensor_set(t, buf.data(), 0, buf.size());
        return;
    }
    GGML_ABORT("unsupported tensor type for fill_random: %s", ggml_type_name(t->type));
}

void fill_zero(ggml_tensor * t) {
    std::vector<uint8_t> buf(ggml_nbytes(t), 0);
    ggml_backend_tensor_set(t, buf.data(), 0, buf.size());
}

// build: creates the op(s) in ctx and returns the output tensor
// zero_names: leaf tensors that are filled with zeros instead of random data (masks)
bench_result bench_graph(ggml_backend_t backend, const std::function<ggml_tensor *(ggml_context *)> & build,
                         const std::vector<std::string> & zero_names, double target_ms = 150.0, int min_runs = 3) {
    bench_result res;

    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 64 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * out = build(ctx);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (!ggml_backend_supports_op(backend, node)) {
            res.reason = std::string("op not supported: ") + ggml_op_desc(node);
            ggml_free(ctx);
            return res;
        }
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        res.reason = "failed to allocate tensors";
        ggml_free(ctx);
        return res;
    }

    std::mt19937 rng(42);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        if (t->op != GGML_OP_NONE || t->view_src != nullptr) {
            continue;
        }
        if (std::find(zero_names.begin(), zero_names.end(), t->name) != zero_names.end()) {
            fill_zero(t);
        } else {
            fill_random(t, rng);
        }
    }

    // warm up: first runs include lazy initialization inside the backend
    for (int i = 0; i < 2; i++) {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            res.reason = "graph compute failed";
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return res;
        }
    }
    ggml_backend_synchronize(backend);

    int64_t total_us = 0;
    int n_runs = 0;
    while (n_runs < min_runs || total_us < target_ms * 1000) {
        const int64_t t0 = ggml_time_us();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        total_us += ggml_time_us() - t0;
        n_runs++;
        if (n_runs >= 1000) {
            break;
        }
    }

    res.supported  = true;
    res.us_per_run = (double) total_us / n_runs;
    res.n_runs     = n_runs;

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

// weight matmul: src0 [k, m] of the weight type times src1 [k, n] f32
bench_result bench_matmul(ggml_backend_t backend, ggml_type type, int64_t k, int64_t m, int64_t n) {
    return bench_graph(backend, [&](ggml_context * ctx) {
        ggml_tensor * w = ggml_new_tensor_2d(ctx, type, k, m);
        ggml_set_name(w, "w");
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
        ggml_set_name(x, "x");
        return ggml_mul_mat(ctx, w, x);
    }, {});
}

// attention over a KV cache of n_kv entries, with flash attention or through the explicit path
bench_result bench_attn(ggml_backend_t backend, bool flash, ggml_type type_kv, int hd, int n_head, int n_head_kv, int n_kv, int n_q) {
    const float scale = 1.0f / std::sqrt((float) hd);
    return bench_graph(backend, [&](ggml_context * ctx) {
        ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, n_q, n_head);
        ggml_set_name(q, "q");
        ggml_tensor * k = ggml_new_tensor_3d(ctx, type_kv, hd, n_kv, n_head_kv);
        ggml_set_name(k, "k");
        ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_q);
        ggml_set_name(mask, "mask");

        if (flash) {
            ggml_tensor * v = ggml_new_tensor_3d(ctx, type_kv, hd, n_kv, n_head_kv);
            ggml_set_name(v, "v");
            ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
            ggml_set_name(out, "out");
            return out;
        }

        // the non-flash path keeps V transposed in the cache: [n_kv, hd, n_head_kv]
        ggml_tensor * vt = ggml_new_tensor_3d(ctx, type_kv, n_kv, hd, n_head_kv);
        ggml_set_name(vt, "vt");
        ggml_tensor * kq = ggml_mul_mat(ctx, k, q);                        // [n_kv, n_q, n_head]
        kq = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);
        ggml_tensor * out = ggml_mul_mat(ctx, vt, kq);                     // [hd, n_q, n_head]
        ggml_set_name(out, "out");
        return out;
    }, { "mask" });
}

std::string now_string() {
    const std::time_t t = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    return buf;
}

std::string attn_key(int hd, ggml_type type_kv) {
    return "hd" + std::to_string(hd) + "/" + ggml_type_name(type_kv);
}

} // namespace

//
// measurement of one device
//

fit_advisor_device_measurements fit_advisor_measure_device(ggml_backend_dev_t dev, const fit_advisor_measure_options & opts) {
    fit_advisor_device_measurements m;

    const bool is_cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    const int n_threads = is_cpu ? (opts.n_threads > 0 ? opts.n_threads : common_cpu_get_num_math()) : 0;

    m.fingerprint = fit_advisor_fingerprint(dev, n_threads);
    m.measured_at = now_string();

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) {
        LOG_ERR("%s: failed to initialize backend for device %s\n", __func__, m.fingerprint.name.c_str());
        return m;
    }
    if (is_cpu) {
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }

    LOG_INF("%s: measuring %s (%s)%s\n", __func__, m.fingerprint.name.c_str(), m.fingerprint.description.c_str(),
        is_cpu ? (" with " + std::to_string(n_threads) + " threads").c_str() : "");

    // matmul per weight type: two weight sizes at batch 1 give slope (bandwidth) and intercept (overhead)
    // the weights are sized in bytes so that the larger one exceeds any on-chip cache and measures memory bandwidth
    constexpr int64_t k = 4096;
    size_t dev_free  = 0;
    size_t dev_total = 0;
    ggml_backend_dev_memory(dev, &dev_free, &dev_total);
    const size_t bytes_small = 64ull * 1024 * 1024;
    const size_t bytes_large = std::min<size_t>(512ull * 1024 * 1024, dev_free > 0 ? dev_free / 4 : 512ull * 1024 * 1024);
    for (const ggml_type type : opts.weight_types) {
        fit_advisor_matmul_rate r;
        r.n_batch_pp = opts.n_batch_pp;
        if (ggml_quantize_requires_imatrix(type)) {
            LOG_WRN("%s: skipping %s, it needs an importance matrix to quantize test data\n", __func__, ggml_type_name(type));
            m.matmul[ggml_type_name(type)] = r;
            continue;
        }
        const size_t  row_bytes = ggml_row_size(type, k);
        const int64_t m1 = (int64_t) (bytes_small / row_bytes);
        const int64_t m2 = (int64_t) (bytes_large / row_bytes);
        r.bytes_small = row_bytes * m1;
        r.bytes_large = row_bytes * m2;

        const bench_result b1 = bench_matmul(backend, type, k, m1, 1);
        const bench_result b2 = bench_matmul(backend, type, k, m2, 1);
        const bench_result bp = bench_matmul(backend, type, k, m1, opts.n_batch_pp);
        if (!b1.supported || !b2.supported) {
            LOG_WRN("%s: matmul %s not measurable: %s\n", __func__, ggml_type_name(type), (b1.supported ? b2 : b1).reason.c_str());
            m.matmul[ggml_type_name(type)] = r;
            continue;
        }
        const double bytes1 = (double) r.bytes_small;
        const double bytes2 = (double) r.bytes_large;
        const double dt_us  = b2.us_per_run - b1.us_per_run;
        r.supported = true;
        if (dt_us > 0) {
            r.bytes_per_s = (bytes2 - bytes1) / (dt_us * 1e-6);
            r.overhead_us = std::max(0.0, b1.us_per_run - bytes1 / r.bytes_per_s * 1e6);
        } else {
            // noise dominated, fall back to the throughput of the larger size
            r.bytes_per_s = bytes2 / (b2.us_per_run * 1e-6);
            r.overhead_us = 0;
        }
        if (bp.supported) {
            r.gflops_pp = 2.0 * k * m1 * opts.n_batch_pp / (bp.us_per_run * 1e-6) / 1e9;
        }
        m.matmul[ggml_type_name(type)] = r;
        if (opts.verbose) {
            LOG_INF("%s:   matmul %-6s tg %7.1f GB/s, overhead %6.1f us, pp %7.0f GFLOPS (%d/%d/%d runs, %zu/%zu MiB)\n", __func__,
                ggml_type_name(type), r.bytes_per_s / 1e9, r.overhead_us, r.gflops_pp, b1.n_runs, b2.n_runs, bp.n_runs,
                r.bytes_small / (1024 * 1024), r.bytes_large / (1024 * 1024));
        }
    }

    // attention per KV type
    for (const ggml_type type_kv : opts.kv_types) {
        fit_advisor_attn_rate r;
        r.n_kv       = opts.n_kv;
        r.n_batch_pp = opts.n_batch_attn;
        const double kv_bytes = 2.0 * ggml_row_size(type_kv, opts.head_size) * opts.n_kv * opts.n_head_kv;

        const bench_result fa1 = bench_attn(backend, true,  type_kv, opts.head_size, opts.n_head, opts.n_head_kv, opts.n_kv, 1);
        const bench_result nf1 = bench_attn(backend, false, type_kv, opts.head_size, opts.n_head, opts.n_head_kv, opts.n_kv, 1);
        const bench_result fap = bench_attn(backend, true,  type_kv, opts.head_size, opts.n_head, opts.n_head_kv, opts.n_kv, opts.n_batch_attn);
        const bench_result nfp = bench_attn(backend, false, type_kv, opts.head_size, opts.n_head, opts.n_head_kv, opts.n_kv, opts.n_batch_attn);

        r.supported_fa   = fa1.supported;
        r.supported_nofa = nf1.supported;
        if (fa1.supported) {
            r.kv_bytes_per_s_fa = kv_bytes / (fa1.us_per_run * 1e-6);
        }
        if (nf1.supported) {
            r.kv_bytes_per_s_nofa = kv_bytes / (nf1.us_per_run * 1e-6);
        }
        r.us_pp_fa   = fap.supported ? fap.us_per_run : 0;
        r.us_pp_nofa = nfp.supported ? nfp.us_per_run : 0;
        m.attn[attn_key(opts.head_size, type_kv)] = r;
        if (opts.verbose) {
            LOG_INF("%s:   attn %-10s fa: tg %7.1f GB/s pp %8.0f us | no-fa: tg %7.1f GB/s pp %8.0f us%s%s\n", __func__,
                attn_key(opts.head_size, type_kv).c_str(),
                r.kv_bytes_per_s_fa / 1e9, r.us_pp_fa, r.kv_bytes_per_s_nofa / 1e9, r.us_pp_nofa,
                fa1.supported ? "" : (" [fa: " + fa1.reason + "]").c_str(),
                nf1.supported ? "" : (" [no-fa: " + nf1.reason + "]").c_str());
        }
    }

    // copies: one large transfer each way, then many small ones for latency
    if (opts.measure_copy) {
        constexpr size_t big_bytes   = 256ull * 1024 * 1024;
        constexpr size_t small_bytes = 4096;

        ggml_init_params ip = { ggml_tensor_overhead() * 4, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * big   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, big_bytes / sizeof(float));
        ggml_tensor * small = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, small_bytes / sizeof(float));
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buf) {
            std::vector<float> host(big_bytes / sizeof(float), 1.0f);

            ggml_backend_tensor_set(big, host.data(), 0, big_bytes); // warm up
            ggml_backend_synchronize(backend);

            int64_t t0 = ggml_time_us();
            ggml_backend_tensor_set(big, host.data(), 0, big_bytes);
            ggml_backend_synchronize(backend);
            int64_t t1 = ggml_time_us();
            m.copy.h2d_gb_s = big_bytes / ((t1 - t0) * 1e-6) / 1e9;

            t0 = ggml_time_us();
            ggml_backend_tensor_get(big, host.data(), 0, big_bytes);
            ggml_backend_synchronize(backend);
            t1 = ggml_time_us();
            m.copy.d2h_gb_s = big_bytes / ((t1 - t0) * 1e-6) / 1e9;

            constexpr int n_small = 200;
            t0 = ggml_time_us();
            for (int i = 0; i < n_small; i++) {
                ggml_backend_tensor_set(small, host.data(), 0, small_bytes);
                ggml_backend_synchronize(backend);
            }
            t1 = ggml_time_us();
            m.copy.latency_us = (double) (t1 - t0) / n_small;

            ggml_backend_buffer_free(buf);
        } else {
            LOG_WRN("%s: could not allocate copy test buffers on %s\n", __func__, m.fingerprint.name.c_str());
        }
        ggml_free(ctx);
        if (opts.verbose) {
            LOG_INF("%s:   copy h2d %6.2f GB/s, d2h %6.2f GB/s, small transfer %6.1f us\n", __func__,
                m.copy.h2d_gb_s, m.copy.d2h_gb_s, m.copy.latency_us);
        }
    }

    ggml_backend_free(backend);
    return m;
}

//
// cache
//

static json fingerprint_to_json(const fit_advisor_device_fingerprint & fp) {
    return {
        { "name",         fp.name },
        { "description",  fp.description },
        { "device_id",    fp.device_id },
        { "backend",      fp.backend },
        { "total_memory", fp.total_memory },
        { "build_commit", fp.build_commit },
        { "n_threads",    fp.n_threads },
    };
}

static fit_advisor_device_fingerprint fingerprint_from_json(const json & j) {
    fit_advisor_device_fingerprint fp;
    fp.name         = j.value("name", "");
    fp.description  = j.value("description", "");
    fp.device_id    = j.value("device_id", "");
    fp.backend      = j.value("backend", "");
    fp.total_memory = j.value("total_memory", (int64_t) 0);
    fp.build_commit = j.value("build_commit", "");
    fp.n_threads    = j.value("n_threads", 0);
    return fp;
}

static json device_to_json(const fit_advisor_device_measurements & m) {
    json j;
    j["fingerprint"] = fingerprint_to_json(m.fingerprint);
    j["measured_at"] = m.measured_at;
    for (const auto & [type, r] : m.matmul) {
        j["matmul"][type] = {
            { "supported",   r.supported },
            { "bytes_per_s", r.bytes_per_s },
            { "overhead_us", r.overhead_us },
            { "gflops_pp",   r.gflops_pp },
            { "n_batch_pp",  r.n_batch_pp },
            { "bytes_small", r.bytes_small },
            { "bytes_large", r.bytes_large },
        };
    }
    for (const auto & [key, r] : m.attn) {
        j["attn"][key] = {
            { "supported_fa",        r.supported_fa },
            { "supported_nofa",      r.supported_nofa },
            { "kv_bytes_per_s_fa",   r.kv_bytes_per_s_fa },
            { "kv_bytes_per_s_nofa", r.kv_bytes_per_s_nofa },
            { "us_pp_fa",            r.us_pp_fa },
            { "us_pp_nofa",          r.us_pp_nofa },
            { "n_kv",                r.n_kv },
            { "n_batch_pp",          r.n_batch_pp },
        };
    }
    j["copy"] = {
        { "h2d_gb_s",   m.copy.h2d_gb_s },
        { "d2h_gb_s",   m.copy.d2h_gb_s },
        { "latency_us", m.copy.latency_us },
    };
    return j;
}

static fit_advisor_device_measurements device_from_json(const json & j) {
    fit_advisor_device_measurements m;
    m.fingerprint = fingerprint_from_json(j.at("fingerprint"));
    m.measured_at = j.value("measured_at", "");
    if (j.contains("matmul")) {
        for (const auto & [type, r] : j.at("matmul").items()) {
            fit_advisor_matmul_rate mr;
            mr.supported   = r.value("supported", false);
            mr.bytes_per_s = r.value("bytes_per_s", 0.0);
            mr.overhead_us = r.value("overhead_us", 0.0);
            mr.gflops_pp   = r.value("gflops_pp", 0.0);
            mr.n_batch_pp  = r.value("n_batch_pp", 0);
            mr.bytes_small = r.value("bytes_small", (size_t) 0);
            mr.bytes_large = r.value("bytes_large", (size_t) 0);
            m.matmul[type] = mr;
        }
    }
    if (j.contains("attn")) {
        for (const auto & [key, r] : j.at("attn").items()) {
            fit_advisor_attn_rate ar;
            ar.supported_fa        = r.value("supported_fa", false);
            ar.supported_nofa      = r.value("supported_nofa", false);
            ar.kv_bytes_per_s_fa   = r.value("kv_bytes_per_s_fa", 0.0);
            ar.kv_bytes_per_s_nofa = r.value("kv_bytes_per_s_nofa", 0.0);
            ar.us_pp_fa            = r.value("us_pp_fa", 0.0);
            ar.us_pp_nofa          = r.value("us_pp_nofa", 0.0);
            ar.n_kv                = r.value("n_kv", 0);
            ar.n_batch_pp          = r.value("n_batch_pp", 0);
            m.attn[key] = ar;
        }
    }
    if (j.contains("copy")) {
        const auto & c = j.at("copy");
        m.copy.h2d_gb_s   = c.value("h2d_gb_s", 0.0);
        m.copy.d2h_gb_s   = c.value("d2h_gb_s", 0.0);
        m.copy.latency_us = c.value("latency_us", 0.0);
    }
    return m;
}

bool fit_advisor_device_measurements::has_attn(int head_size, ggml_type type_kv) const {
    return attn.count(attn_key(head_size, type_kv)) > 0;
}

std::string fit_advisor_measurements::default_path() {
    return fs_get_cache_directory() + "fit-advisor-measurements.json";
}

bool fit_advisor_measurements::load(const std::string & path) {
    std::ifstream f(path);
    if (!f.good()) {
        return false;
    }
    try {
        const json j = json::parse(f);
        if (j.value("version", 0) != 1) {
            LOG_WRN("%s: ignoring %s, unknown version\n", __func__, path.c_str());
            return false;
        }
        for (const auto & [key, jd] : j.at("devices").items()) {
            devices[key] = device_from_json(jd);
        }
        return true;
    } catch (const std::exception & e) {
        LOG_WRN("%s: ignoring %s: %s\n", __func__, path.c_str(), e.what());
        return false;
    }
}

std::string fit_advisor_measurements::to_json() const {
    json j;
    j["version"] = 1;
    j["devices"] = json::object();
    for (const auto & [key, m] : devices) {
        j["devices"][key] = device_to_json(m);
    }
    return j.dump(2);
}

bool fit_advisor_measurements::save(const std::string & path) const {
    fs_create_directory_with_parents(fs_get_cache_directory());
    std::ofstream f(path);
    if (!f.good()) {
        LOG_WRN("%s: cannot write %s\n", __func__, path.c_str());
        return false;
    }
    f << to_json() << "\n";
    return true;
}

const fit_advisor_device_measurements * fit_advisor_measurements::find(ggml_backend_dev_t dev, int n_threads) const {
    const bool is_cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    const std::string key = fit_advisor_fingerprint(dev, is_cpu ? (n_threads > 0 ? n_threads : common_cpu_get_num_math()) : 0).key();
    auto it = devices.find(key);
    return it == devices.end() ? nullptr : &it->second;
}

const fit_advisor_device_measurements & fit_advisor_measurements::ensure(ggml_backend_dev_t dev, const fit_advisor_measure_options & opts, bool force) {
    const bool is_cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    const int n_threads = is_cpu ? (opts.n_threads > 0 ? opts.n_threads : common_cpu_get_num_math()) : 0;
    const std::string key = fit_advisor_fingerprint(dev, n_threads).key();

    auto it = devices.find(key);
    fit_advisor_measure_options todo = opts;

    if (it != devices.end() && !force) {
        // measure only what the cached entry lacks
        const fit_advisor_device_measurements & have = it->second;
        todo.weight_types.clear();
        for (const ggml_type type : opts.weight_types) {
            if (!have.has_matmul(type)) {
                todo.weight_types.push_back(type);
            }
        }
        todo.kv_types.clear();
        for (const ggml_type type : opts.kv_types) {
            if (!have.has_attn(opts.head_size, type)) {
                todo.kv_types.push_back(type);
            }
        }
        todo.measure_copy = have.copy.h2d_gb_s <= 0;
        if (todo.weight_types.empty() && todo.kv_types.empty() && !todo.measure_copy) {
            return have;
        }
        LOG_INF("%s: cached entry for %s lacks %zu weight types, %zu KV types%s, measuring those\n", __func__,
            ggml_backend_dev_name(dev), todo.weight_types.size(), todo.kv_types.size(), todo.measure_copy ? " and copy rates" : "");
    }

    fit_advisor_device_measurements m = fit_advisor_measure_device(dev, todo);

    if (it != devices.end() && !force) {
        fit_advisor_device_measurements & have = it->second;
        for (auto & [type, r] : m.matmul) {
            have.matmul[type] = r;
        }
        for (auto & [k, r] : m.attn) {
            have.attn[k] = r;
        }
        if (todo.measure_copy) {
            have.copy = m.copy;
        }
        have.measured_at = m.measured_at;
    } else {
        devices[key] = std::move(m);
    }
    save(default_path());
    return devices[key];
}

void fit_advisor_measurements_print(const fit_advisor_device_measurements & m) {
    LOG_INF("%s: %s (%s) via %s, measured %s\n", __func__, m.fingerprint.name.c_str(), m.fingerprint.description.c_str(),
        m.fingerprint.backend.c_str(), m.measured_at.c_str());
    for (const auto & [type, r] : m.matmul) {
        if (!r.supported) {
            LOG_INF("%s:   matmul %-6s not supported\n", __func__, type.c_str());
            continue;
        }
        LOG_INF("%s:   matmul %-6s tg %7.1f GB/s, overhead %6.1f us, pp %7.0f GFLOPS at batch %d\n", __func__,
            type.c_str(), r.bytes_per_s / 1e9, r.overhead_us, r.gflops_pp, r.n_batch_pp);
    }
    for (const auto & [key, r] : m.attn) {
        LOG_INF("%s:   attn %-10s n_kv %d: fa tg %7.1f GB/s pp %8.0f us | no-fa tg %7.1f GB/s pp %8.0f us%s%s\n", __func__,
            key.c_str(), r.n_kv, r.kv_bytes_per_s_fa / 1e9, r.us_pp_fa, r.kv_bytes_per_s_nofa / 1e9, r.us_pp_nofa,
            r.supported_fa ? "" : " [fa unsupported]", r.supported_nofa ? "" : " [no-fa unsupported]");
    }
    LOG_INF("%s:   copy h2d %6.2f GB/s, d2h %6.2f GB/s, small transfer %6.1f us\n", __func__,
        m.copy.h2d_gb_s, m.copy.d2h_gb_s, m.copy.latency_us);
}
