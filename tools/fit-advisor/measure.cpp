#include "measure.h"

#include "build-info.h"
#include "common.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cinttypes>
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
    // the build commit is recorded but not part of the key: kernel changes are rare enough that --remeasure covers them,
    // and keying on the commit re-measured everything on every push
    std::ostringstream ss;
    ss << backend << "|" << name << "|" << description << "|" << device_id << "|" << total_memory;
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

// fill a leaf tensor with random data of its type
// only a block of rows is generated and converted, then tiled across the tensor: kernel speed does not depend on
// the values, and the search-based quantizers (IQ types) would take minutes on a 512 MiB weight
void fill_random(ggml_tensor * t, std::mt19937 & rng) {
    const int64_t n_per_row = t->ne[0];
    const int64_t nrows     = ggml_nelements(t) / n_per_row;
    // the importance-matrix quantizers are search based and slow, keep their block small
    const int64_t nrows_blk = std::min<int64_t>(nrows, ggml_quantize_requires_imatrix(t->type) ? 4 : 64);
    const size_t  row_bytes = ggml_row_size(t->type, n_per_row);
    const size_t  blk_bytes = row_bytes * nrows_blk;

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(n_per_row * nrows_blk);
    for (auto & x : data) {
        x = dist(rng);
    }

    std::vector<uint8_t> blk(blk_bytes);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(blk.data(), data.data(), blk_bytes);
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(data.data(), (ggml_fp16_t *) blk.data(), data.size());
    } else if (t->type == GGML_TYPE_BF16) {
        ggml_fp32_to_bf16_row(data.data(), (ggml_bf16_t *) blk.data(), data.size());
    } else if (ggml_is_quantized(t->type)) {
        // types that need an importance matrix get a flat one: it only steers codebook choice, not kernel speed
        std::vector<float> imatrix;
        const float * im = nullptr;
        if (ggml_quantize_requires_imatrix(t->type)) {
            imatrix.assign(n_per_row, 1.0f);
            im = imatrix.data();
        }
        ggml_quantize_chunk(t->type, data.data(), blk.data(), 0, nrows_blk, n_per_row, im);
    } else {
        GGML_ABORT("unsupported tensor type for fill_random: %s", ggml_type_name(t->type));
    }

    std::vector<uint8_t> buf(ggml_nbytes(t));
    for (size_t off = 0; off < buf.size(); off += blk_bytes) {
        std::memcpy(buf.data() + off, blk.data(), std::min(blk_bytes, buf.size() - off));
    }
    ggml_backend_tensor_set(t, buf.data(), 0, buf.size());
}

void fill_zero(ggml_tensor * t) {
    std::vector<uint8_t> buf(ggml_nbytes(t), 0);
    ggml_backend_tensor_set(t, buf.data(), 0, buf.size());
}

// build: creates the op(s) in ctx and returns the output tensor
// zero_names: leaf tensors that are filled with zeros instead of random data (masks)
using custom_fill_fn = std::function<bool(ggml_tensor *, std::mt19937 &)>; // true when the tensor was filled

bench_result bench_graph(ggml_backend_t backend, const std::function<ggml_tensor *(ggml_context *)> & build,
                         const std::vector<std::string> & zero_names, double target_ms = 150.0, int min_runs = 3,
                         const custom_fill_fn & custom_fill = nullptr) {
    bench_result res;

    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(512, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * out = build(ctx);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 512, false);
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
        if (custom_fill && custom_fill(t, rng)) {
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

// expert matmul as the MoE layers issue it: weights [k, m, n_expert], activations [k, 1, n_tokens], ids [n_used, n_tokens]
// with distinct random experts per token, so the routing, gather and sort work is part of the measurement
bench_result bench_mul_mat_id(ggml_backend_t backend, ggml_type type, int64_t k, int64_t m, int n_expert, int n_used, int n_tokens) {
    return bench_graph(backend, [&](ggml_context * ctx) {
        ggml_tensor * w = ggml_new_tensor_3d(ctx, type, k, m, n_expert);
        ggml_set_name(w, "w");
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, 1, n_tokens);
        ggml_set_name(x, "x");
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
        ggml_set_name(ids, "ids");
        return ggml_mul_mat_id(ctx, w, x, ids);
    }, {}, 150.0, 3, [&](ggml_tensor * t, std::mt19937 & rng) {
        if (t->type != GGML_TYPE_I32) {
            return false;
        }
        std::vector<int32_t> ids((size_t) ggml_nelements(t));
        std::vector<int32_t> perm(n_expert);
        for (int64_t tok = 0; tok < t->ne[1]; tok++) {
            for (int e = 0; e < n_expert; e++) {
                perm[e] = e;
            }
            for (int j = 0; j < n_used; j++) { // partial Fisher-Yates: n_used distinct experts
                const int r = j + (int) (rng() % (uint32_t) (n_expert - j));
                std::swap(perm[j], perm[r]);
                ids[(size_t) tok * n_used + j] = perm[j];
            }
        }
        ggml_backend_tensor_set(t, ids.data(), 0, ids.size() * sizeof(int32_t));
        return true;
    });
}

// attention over a KV cache of n_kv entries, with flash attention or through the explicit path
bench_result bench_attn(ggml_backend_t backend, bool flash, ggml_type type_kv, int hd, int hdv, int n_head, int n_head_kv, int n_kv, int n_q) {
    const float scale = 1.0f / std::sqrt((float) hd);
    return bench_graph(backend, [&](ggml_context * ctx) {
        ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, n_q, n_head);
        ggml_set_name(q, "q");
        ggml_tensor * k = ggml_new_tensor_3d(ctx, type_kv, hd, n_kv, n_head_kv);
        ggml_set_name(k, "k");
        ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_q);
        ggml_set_name(mask, "mask");

        if (flash) {
            ggml_tensor * v = ggml_new_tensor_3d(ctx, type_kv, hdv, n_kv, n_head_kv);
            ggml_set_name(v, "v");
            ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
            ggml_set_name(out, "out");
            return out;
        }

        // the non-flash path keeps V transposed in the cache: [n_kv, hdv, n_head_kv]
        ggml_tensor * vt = ggml_new_tensor_3d(ctx, type_kv, n_kv, hdv, n_head_kv);
        ggml_set_name(vt, "vt");
        ggml_tensor * kq = ggml_mul_mat(ctx, k, q);                        // [n_kv, n_q, n_head]
        kq = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);
        ggml_tensor * out = ggml_mul_mat(ctx, vt, kq);                     // [hd, n_q, n_head]
        ggml_set_name(out, "out");
        return out;
    }, { "mask" });
}

// a chain of n trivial elementwise ops on a tiny tensor, each consuming the previous result: the slope in n is the
// fixed cost of a graph node on this device (launch or barrier), with next to no kernel work in it
bench_result bench_chain(ggml_backend_t backend, int n_ops) {
    return bench_graph(backend, [&](ggml_context * ctx) {
        constexpr int64_t k = 256;
        ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k);
        ggml_set_name(x, "x");
        ggml_tensor * cur = x;
        for (int i = 0; i < n_ops; i++) {
            cur = ggml_scale(ctx, cur, 1.0001f);
        }
        return cur;
    }, {}, 100.0, 5);
}

std::string now_string() {
    const std::time_t t = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    return buf;
}

std::string attn_key(int hd, int hdv, ggml_type type_kv) {
    return "hd" + std::to_string(hd) + (hdv != hd ? "v" + std::to_string(hdv) : "") + "/" + ggml_type_name(type_kv);
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

    // free memory before the backend exists: what the runtime keeps once the kernels have run is measured against this
    size_t free_before = 0;
    size_t total_dev   = 0;
    ggml_backend_dev_memory(dev, &free_before, &total_dev);

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

    // per-op cost inside a graph: the slope of a chain of tiny ops, so thread start-up and launch setup are excluded
    {
        const bench_result c8  = bench_chain(backend, 8);
        const bench_result c40 = bench_chain(backend, 40);
        if (c8.supported && c40.supported && c40.us_per_run > c8.us_per_run) {
            m.op_overhead_us = (c40.us_per_run - c8.us_per_run) / 32.0;
        } else if (c40.supported) {
            m.op_overhead_us = c40.us_per_run / 40.0;
        }
        const bench_result c1 = bench_chain(backend, 1);
        m.launch_us = c1.supported ? c1.us_per_run : 0;
        LOG_INF("%s:   per-op overhead inside a graph %.1f us, one-op graph compute %.1f us\n", __func__, m.op_overhead_us, m.launch_us);
    }

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
        const size_t  row_bytes = ggml_row_size(type, k);
        const int64_t m1 = (int64_t) (bytes_small / row_bytes);
        const int64_t m2 = (int64_t) (bytes_large / row_bytes);
        r.bytes_small = row_bytes * m1;
        r.bytes_large = row_bytes * m2;
        LOG_INF("%s:   matmul %-6s measuring (%zu and %zu MiB weights, batch 1 and %d) ...\n", __func__,
            ggml_type_name(type), r.bytes_small / (1024 * 1024), r.bytes_large / (1024 * 1024), opts.n_batch_pp);
        const int64_t t_type0 = ggml_time_us();

        const bench_result b1 = bench_matmul(backend, type, k, m1, 1);
        const bench_result b2 = bench_matmul(backend, type, k, m2, 1);
        const bench_result b4 = bench_matmul(backend, type, k, m2, 4);
        const bench_result bp = bench_matmul(backend, type, k, m1, opts.n_batch_pp);
        // the rest of the curve: the large weight below the prompt batch, the small one at it
        for (const int b : opts.batches) {
            if (b == 1 || b == 4 || b >= opts.n_batch_pp) {
                continue;
            }
            const bench_result bb = bench_matmul(backend, type, k, m2, b);
            if (bb.supported) {
                r.points[b] = bb.us_per_run * 1e-6 / (double) r.bytes_large;
            }
        }
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
        } else {
            // noise dominated, fall back to the throughput of the larger size
            r.bytes_per_s = bytes2 / (b2.us_per_run * 1e-6);
        }
        r.overhead_us = m.op_overhead_us;
        if (bp.supported) {
            r.gflops_pp = 2.0 * k * m1 * opts.n_batch_pp / (bp.us_per_run * 1e-6) / 1e9;
            r.s_per_byte_bpp = bp.us_per_run * 1e-6 / bytes1;
        }
        r.s_per_byte_b1 = b2.us_per_run * 1e-6 / bytes2;
        r.points[1] = r.s_per_byte_b1;
        if (b4.supported) {
            r.s_per_byte_b4 = b4.us_per_run * 1e-6 / bytes2;
            r.points[4] = r.s_per_byte_b4;
        }
        if (bp.supported) {
            r.points[opts.n_batch_pp] = r.s_per_byte_bpp;
        }
        m.matmul[ggml_type_name(type)] = r;
        LOG_INF("%s:   matmul %-6s tg %7.1f GB/s, b4 %7.1f GB/s/token, overhead %6.1f us, pp %7.0f GFLOPS (%d/%d/%d/%d runs, %.1f s)\n", __func__,
            ggml_type_name(type), r.bytes_per_s / 1e9, r.s_per_byte_b4 > 0 ? 1.0 / r.s_per_byte_b4 / 4 / 1e9 : 0.0, r.overhead_us, r.gflops_pp,
            b1.n_runs, b2.n_runs, b4.n_runs, bp.n_runs, (ggml_time_us() - t_type0) * 1e-6);
    }

    // expert matmul per expert weight type, with the model's routing geometry and expert shape
    if (opts.n_expert > 0 && opts.n_expert_used > 0 && opts.moe_k > 0 && opts.moe_m > 0) {
        for (const ggml_type type : opts.moe_types) {
            fit_advisor_moe_rate r;
            r.n_expert      = opts.n_expert;
            r.n_expert_used = opts.n_expert_used;
            r.k = opts.moe_k;
            r.m = opts.moe_m;
            r.bytes_total = ggml_row_size(type, r.k) * (size_t) r.m * (size_t) r.n_expert;
            if (dev_free > 0 && r.bytes_total > dev_free / 2) {
                LOG_WRN("%s: moe %s not measurable: the expert stack needs %zu MiB\n", __func__, ggml_type_name(type), r.bytes_total / (1024 * 1024));
                m.moe[ggml_type_name(type)] = r;
                continue;
            }
            std::vector<int> batches = opts.batches;
            if (std::find(batches.begin(), batches.end(), opts.n_batch_pp) == batches.end()) {
                batches.push_back(opts.n_batch_pp);
            }
            LOG_INF("%s:   moe %-6s measuring (%d experts of [%" PRId64 ", %" PRId64 "], %d used, %zu MiB, %zu batch points) ...\n", __func__,
                ggml_type_name(type), r.n_expert, r.k, r.m, r.n_expert_used, r.bytes_total / (1024 * 1024), batches.size());
            const int64_t t_moe0 = ggml_time_us();
            std::string reason;
            for (const int b : batches) {
                if (b > opts.n_batch_pp) {
                    continue;
                }
                const bench_result br = bench_mul_mat_id(backend, type, r.k, r.m, r.n_expert, r.n_expert_used, b);
                if (br.supported) {
                    r.points[b] = br.us_per_run * 1e-6 / (double) r.bytes_total;
                } else {
                    reason = br.reason;
                }
            }
            r.supported = r.points.count(1) > 0;
            if (!r.supported) {
                LOG_WRN("%s: moe %s not measurable: %s\n", __func__, ggml_type_name(type), reason.c_str());
            } else {
                const double touched = (double) r.bytes_total * r.n_expert_used / r.n_expert;
                std::string pts;
                for (const auto & [b, spb] : r.points) {
                    char buf[64];
                    snprintf(buf, sizeof(buf), "%s b%d %.0f us", pts.empty() ? "" : ",", b, spb * r.bytes_total * 1e6);
                    pts += buf;
                }
                LOG_INF("%s:   moe %-6s tg %7.1f GB/s over the %d used experts;%s (%.1f s)\n", __func__, ggml_type_name(type),
                    touched / (r.points[1] * r.bytes_total) / 1e9, r.n_expert_used, pts.c_str(), (ggml_time_us() - t_moe0) * 1e-6);
            }
            m.moe[ggml_type_name(type)] = r;
        }
    }

    // attention per KV type
    for (const ggml_type type_kv : opts.kv_types) {
        fit_advisor_attn_rate r;
        r.n_kv       = opts.n_kv;
        r.n_batch_pp = opts.n_batch_attn;
        const int hdv = opts.head_size_v > 0 ? opts.head_size_v : opts.head_size;
        const double kv_bytes = (ggml_row_size(type_kv, opts.head_size) + ggml_row_size(type_kv, hdv)) * (double) opts.n_kv * opts.n_head_kv;
        LOG_INF("%s:   attn %-10s measuring (n_kv %d, %d heads / %d kv heads, batch 1 and %d, with and without flash attention) ...\n", __func__,
            attn_key(opts.head_size, hdv, type_kv).c_str(), opts.n_kv, opts.n_head, opts.n_head_kv, opts.n_batch_attn);
        const int64_t t_attn0 = ggml_time_us();

        const bench_result fa1 = bench_attn(backend, true,  type_kv, opts.head_size, hdv, opts.n_head, opts.n_head_kv, opts.n_kv, 1);
        const bench_result nf1 = bench_attn(backend, false, type_kv, opts.head_size, hdv, opts.n_head, opts.n_head_kv, opts.n_kv, 1);
        const bench_result fap = bench_attn(backend, true,  type_kv, opts.head_size, hdv, opts.n_head, opts.n_head_kv, opts.n_kv, opts.n_batch_attn);
        const bench_result nfp = bench_attn(backend, false, type_kv, opts.head_size, hdv, opts.n_head, opts.n_head_kv, opts.n_kv, opts.n_batch_attn);

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
        m.attn[attn_key(opts.head_size, hdv, type_kv)] = r;
        LOG_INF("%s:   attn %-10s fa: tg %7.1f GB/s pp %8.0f us | no-fa: tg %7.1f GB/s pp %8.0f us (%.1f s)%s%s\n", __func__,
            attn_key(opts.head_size, hdv, type_kv).c_str(),
            r.kv_bytes_per_s_fa / 1e9, r.us_pp_fa, r.kv_bytes_per_s_nofa / 1e9, r.us_pp_nofa,
            (ggml_time_us() - t_attn0) * 1e-6,
            fa1.supported ? "" : (" [fa: " + fa1.reason + "]").c_str(),
            nf1.supported ? "" : (" [no-fa: " + nf1.reason + "]").c_str());
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
        LOG_INF("%s:   copy h2d %6.2f GB/s, d2h %6.2f GB/s, small transfer %6.1f us\n", __func__,
            m.copy.h2d_gb_s, m.copy.d2h_gb_s, m.copy.latency_us);
    }

    ggml_backend_synchronize(backend);
    ggml_backend_free(backend);

    // with the backend gone every buffer, pool and handle is released; what is still missing from the device's free
    // memory is runtime state the loader never sees (kernel modules loaded on first use, driver bookkeeping)
    if (is_cpu) {
        m.runtime_overhead_bytes = 0;
    } else {
        size_t free_after = 0;
        ggml_backend_dev_memory(dev, &free_after, &total_dev);
        m.runtime_overhead_bytes = free_before > free_after ? (int64_t) (free_before - free_after) : 0;
        LOG_INF("%s:   runtime overhead after the kernels ran %.0f MiB (free %.0f -> %.0f MiB)\n", __func__,
            m.runtime_overhead_bytes / (1024.0 * 1024), free_before / (1024.0 * 1024), free_after / (1024.0 * 1024));
    }
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
    j["op_overhead_us"] = m.op_overhead_us;
    j["launch_us"] = m.launch_us;
    j["runtime_overhead_bytes"] = m.runtime_overhead_bytes;
    for (const auto & [type, r] : m.matmul) {
        j["matmul"][type] = {
            { "supported",   r.supported },
            { "bytes_per_s", r.bytes_per_s },
            { "overhead_us", r.overhead_us },
            { "gflops_pp",   r.gflops_pp },
            { "n_batch_pp",  r.n_batch_pp },
            { "bytes_small", r.bytes_small },
            { "bytes_large", r.bytes_large },
            { "s_per_byte_b1",  r.s_per_byte_b1 },
            { "s_per_byte_b4",  r.s_per_byte_b4 },
            { "s_per_byte_bpp", r.s_per_byte_bpp },
        };
        for (const auto & [b, spb] : r.points) {
            j["matmul"][type]["points"][std::to_string(b)] = spb;
        }
    }
    for (const auto & [type, r] : m.moe) {
        j["moe"][type] = {
            { "supported",     r.supported },
            { "n_expert",      r.n_expert },
            { "n_expert_used", r.n_expert_used },
            { "k",             r.k },
            { "m",             r.m },
            { "bytes_total",   r.bytes_total },
        };
        for (const auto & [b, spb] : r.points) {
            j["moe"][type]["points"][std::to_string(b)] = spb;
        }
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
    m.op_overhead_us = j.value("op_overhead_us", 0.0);
    m.launch_us = j.value("launch_us", 0.0);
    m.runtime_overhead_bytes = j.value("runtime_overhead_bytes", (int64_t) -1);
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
            mr.s_per_byte_b1  = r.value("s_per_byte_b1", 0.0);
            mr.s_per_byte_b4  = r.value("s_per_byte_b4", 0.0);
            mr.s_per_byte_bpp = r.value("s_per_byte_bpp", 0.0);
            if (r.contains("points")) {
                for (const auto & [b, spb] : r.at("points").items()) {
                    mr.points[std::stoi(b)] = spb.get<double>();
                }
            }
            m.matmul[type] = mr;
        }
    }
    if (j.contains("moe")) {
        for (const auto & [type, r] : j.at("moe").items()) {
            fit_advisor_moe_rate mr;
            mr.supported     = r.value("supported", false);
            mr.n_expert      = r.value("n_expert", 0);
            mr.n_expert_used = r.value("n_expert_used", 0);
            mr.k             = r.value("k", (int64_t) 0);
            mr.m             = r.value("m", (int64_t) 0);
            mr.bytes_total   = r.value("bytes_total", (size_t) 0);
            if (r.contains("points")) {
                for (const auto & [b, spb] : r.at("points").items()) {
                    mr.points[std::stoi(b)] = spb.get<double>();
                }
            }
            m.moe[type] = mr;
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

bool fit_advisor_device_measurements::has_matmul_curve(ggml_type type, const std::vector<int> & batches) const {
    const auto it = matmul.find(ggml_type_name(type));
    if (it == matmul.end()) {
        return false;
    }
    if (!it->second.supported) {
        return true; // measured and found unsupported, nothing more to learn
    }
    for (const int b : batches) {
        if (it->second.points.count(b) == 0) {
            return false;
        }
    }
    return true;
}

bool fit_advisor_device_measurements::has_moe(ggml_type type, int n_expert, int n_expert_used, const std::vector<int> & batches) const {
    const auto it = moe.find(ggml_type_name(type));
    if (it == moe.end() || it->second.n_expert != n_expert || it->second.n_expert_used != n_expert_used) {
        return false;
    }
    if (!it->second.supported) {
        return true;
    }
    for (const int b : batches) {
        if (it->second.points.count(b) == 0) {
            return false;
        }
    }
    return true;
}

bool fit_advisor_device_measurements::has_attn(int head_size, int head_size_v, ggml_type type_kv) const {
    return attn.count(attn_key(head_size, head_size_v > 0 ? head_size_v : head_size, type_kv)) > 0;
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
        if (j.value("version", 0) != 5) {
            LOG_WRN("%s: ignoring %s, unknown version\n", __func__, path.c_str());
            return false;
        }
        for (const auto & [key, jd] : j.at("devices").items()) {
            devices[key] = device_from_json(jd);
        }
        if (j.contains("pairs")) {
            for (const auto & [key, jp] : j.at("pairs").items()) {
                fit_advisor_pair_rate pr;
                pr.latency_us       = jp.value("latency_us", 0.0);
                pr.gb_s             = jp.value("gb_s", 0.0);
                pr.split_us_b1      = jp.value("split_us_b1", 0.0);
                pr.split_us_bpp     = jp.value("split_us_bpp", 0.0);
                pr.split_bytes_bpp  = jp.value("split_bytes_bpp", (size_t) 0);
                pr.split_n_batch_pp = jp.value("split_n_batch_pp", 0);
                pairs[key] = pr;
            }
        }
        return true;
    } catch (const std::exception & e) {
        LOG_WRN("%s: ignoring %s: %s\n", __func__, path.c_str(), e.what());
        return false;
    }
}

std::string fit_advisor_measurements::to_json() const {
    json j;
    j["version"] = 5;
    j["devices"] = json::object();
    for (const auto & [key, m] : devices) {
        j["devices"][key] = device_to_json(m);
    }
    j["pairs"] = json::object();
    for (const auto & [key, r] : pairs) {
        j["pairs"][key] = {
            { "latency_us", r.latency_us }, { "gb_s", r.gb_s },
            { "split_us_b1", r.split_us_b1 }, { "split_us_bpp", r.split_us_bpp },
            { "split_bytes_bpp", r.split_bytes_bpp }, { "split_n_batch_pp", r.split_n_batch_pp },
        };
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
        std::vector<int> batches = opts.batches;
        if (std::find(batches.begin(), batches.end(), opts.n_batch_pp) == batches.end()) {
            batches.push_back(opts.n_batch_pp);
        }
        todo.weight_types.clear();
        for (const ggml_type type : opts.weight_types) {
            if (!have.has_matmul_curve(type, batches)) {
                todo.weight_types.push_back(type);
            }
        }
        todo.moe_types.clear();
        for (const ggml_type type : opts.moe_types) {
            if (!have.has_moe(type, opts.n_expert, opts.n_expert_used, batches)) {
                todo.moe_types.push_back(type);
            }
        }
        todo.kv_types.clear();
        for (const ggml_type type : opts.kv_types) {
            if (!have.has_attn(opts.head_size, opts.head_size_v, type)) {
                todo.kv_types.push_back(type);
            }
        }
        todo.measure_copy = have.copy.h2d_gb_s <= 0;
        const bool need_launch = have.launch_us <= 0;
        if (have.runtime_overhead_bytes < 0) {
            // the runtime overhead is the memory the whole kernel set leaves behind, so measure everything again
            todo = opts;
        } else if (todo.weight_types.empty() && todo.moe_types.empty() && todo.kv_types.empty() && !todo.measure_copy && !need_launch) {
            return have;
        }
        LOG_INF("%s: cached entry for %s lacks %zu weight types, %zu expert types, %zu KV types%s, measuring those\n", __func__,
            ggml_backend_dev_name(dev), todo.weight_types.size(), todo.moe_types.size(), todo.kv_types.size(), todo.measure_copy ? " and copy rates" : "");
    }

    fit_advisor_device_measurements m = fit_advisor_measure_device(dev, todo);

    if (it != devices.end() && !force) {
        fit_advisor_device_measurements & have = it->second;
        for (auto & [type, r] : m.matmul) {
            have.matmul[type] = r;
        }
        for (auto & [type, r] : m.moe) {
            have.moe[type] = r;
        }
        for (auto & [k, r] : m.attn) {
            have.attn[k] = r;
        }
        if (todo.measure_copy) {
            have.copy = m.copy;
        }
        have.op_overhead_us = m.op_overhead_us;
        have.launch_us      = m.launch_us;
        have.measured_at    = m.measured_at;
        // a partial run loads only part of the kernel set, keep the largest footprint seen
        have.runtime_overhead_bytes = std::max(have.runtime_overhead_bytes, m.runtime_overhead_bytes);
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
        std::string curve; // per-token weight throughput at every measured batch
        for (const auto & [b, spb] : r.points) {
            char buf[48];
            snprintf(buf, sizeof(buf), "%sb%d %.0f", curve.empty() ? "" : " ", b, spb > 0 ? 1.0 / spb / b / 1e9 : 0.0);
            curve += buf;
        }
        LOG_INF("%s:   matmul %-6s tg %7.1f GB/s, overhead %6.1f us, pp %7.0f GFLOPS at batch %d; GB/s per token: %s\n", __func__,
            type.c_str(), r.bytes_per_s / 1e9, r.overhead_us, r.gflops_pp, r.n_batch_pp, curve.c_str());
    }
    for (const auto & [type, r] : m.moe) {
        if (!r.supported) {
            LOG_INF("%s:   moe    %-6s not measured\n", __func__, type.c_str());
            continue;
        }
        std::string curve; // microseconds per ubatch for the stack
        for (const auto & [b, spb] : r.points) {
            char buf[48];
            snprintf(buf, sizeof(buf), "%sb%d %.0f", curve.empty() ? "" : " ", b, spb * r.bytes_total * 1e6);
            curve += buf;
        }
        LOG_INF("%s:   moe    %-6s %d experts [%" PRId64 "x%" PRId64 "], %d used, %zu MiB; us per ubatch: %s\n", __func__,
            type.c_str(), r.n_expert, r.k, r.m, r.n_expert_used, r.bytes_total / (1024 * 1024), curve.c_str());
    }
    for (const auto & [key, r] : m.attn) {
        LOG_INF("%s:   attn %-10s n_kv %d: fa tg %7.1f GB/s pp %8.0f us | no-fa tg %7.1f GB/s pp %8.0f us%s%s\n", __func__,
            key.c_str(), r.n_kv, r.kv_bytes_per_s_fa / 1e9, r.us_pp_fa, r.kv_bytes_per_s_nofa / 1e9, r.us_pp_nofa,
            r.supported_fa ? "" : " [fa unsupported]", r.supported_nofa ? "" : " [no-fa unsupported]");
    }
    LOG_INF("%s:   per-op overhead %.1f us, graph launch %.1f us; copy h2d %6.2f GB/s, d2h %6.2f GB/s, small transfer %6.1f us\n", __func__,
        m.op_overhead_us, m.launch_us, m.copy.h2d_gb_s, m.copy.d2h_gb_s, m.copy.latency_us);
    if (m.runtime_overhead_bytes >= 0) {
        LOG_INF("%s:   runtime overhead %.0f MiB (kernel modules and driver state outside every buffer)\n", __func__,
            m.runtime_overhead_bytes / (1024.0 * 1024));
    }
}

// time one scheduler round: a chain of n_ops tiny matmuls whose weights live on backend a, except every 4th which lives
// on backend b; with n_b = 0 the whole chain is on a, with n_b = n_ops on b. returns microseconds per graph compute
static double bench_sched_chain(ggml_backend_t ba, ggml_backend_t bb, int n_ops, int every, int batch, bool all_on_b) {
    constexpr int64_t k = 256;
    ggml_init_params ip = { ggml_tensor_overhead() * (size_t) (n_ops + 8), nullptr, true };
    ggml_context * ctx_a = ggml_init(ip);
    ggml_context * ctx_b = ggml_init(ip);
    ggml_context * ctx_x = ggml_init(ip);
    ggml_context * ctx_g = ggml_init({ ggml_tensor_overhead() * (size_t) (n_ops + 8) + ggml_graph_overhead_custom(512, false), nullptr, true });

    std::vector<ggml_tensor *> w(n_ops);
    for (int i = 0; i < n_ops; i++) {
        const bool on_b = all_on_b || (every > 0 && i % every == every - 1);
        w[i] = ggml_new_tensor_2d(on_b ? ctx_b : ctx_a, GGML_TYPE_F16, k, k);
        ggml_format_name(w[i], "w%d", i);
    }
    ggml_tensor * x = ggml_new_tensor_2d(ctx_x, GGML_TYPE_F32, k, batch);
    ggml_set_name(x, "x");

    ggml_backend_buffer_t buf_a = ggml_backend_alloc_ctx_tensors(ctx_a, ba);
    ggml_backend_buffer_t buf_b = ggml_backend_alloc_ctx_tensors(ctx_b, bb);
    ggml_backend_buffer_t buf_x = ggml_backend_alloc_ctx_tensors(ctx_x, ba);
    double ret = -1;
    if ((buf_a || ggml_get_first_tensor(ctx_a) == nullptr) && (buf_b || ggml_get_first_tensor(ctx_b) == nullptr) && buf_x) {
        if (buf_a) ggml_backend_buffer_set_usage(buf_a, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if (buf_b) ggml_backend_buffer_set_usage(buf_b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::mt19937 rng(7);
        for (int i = 0; i < n_ops; i++) {
            fill_random(w[i], rng);
        }
        fill_random(x, rng);

        ggml_tensor * cur = x;
        for (int i = 0; i < n_ops; i++) {
            cur = ggml_mul_mat(ctx_g, w[i], cur);
        }
        ggml_cgraph * gf = ggml_new_graph_custom(ctx_g, 512, false);
        ggml_build_forward_expand(gf, cur);

        // the scheduler wants the CPU backend last; op offload is off so an op runs where its weight lives at any batch
        std::vector<ggml_backend_t> backends;
        ggml_backend_t extra_cpu = nullptr;
        for (ggml_backend_t b : { ba, bb }) {
            if (!ggml_backend_is_cpu(b)) {
                backends.push_back(b);
            }
        }
        if (ggml_backend_is_cpu(bb)) {
            backends.push_back(bb);
        } else if (ggml_backend_is_cpu(ba)) {
            backends.push_back(ba);
        } else {
            extra_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
            backends.push_back(extra_cpu);
        }
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(), 512, false, false);
        if (ggml_backend_sched_reserve(sched, gf)) {
            for (int i = 0; i < 3; i++) {
                ggml_backend_sched_graph_compute(sched, gf);
            }
            ggml_backend_sched_synchronize(sched);
            int n = 0;
            int64_t total = 0;
            while (n < 10 || total < 200 * 1000) {
                const int64_t t0 = ggml_time_us();
                ggml_backend_sched_graph_compute(sched, gf);
                ggml_backend_sched_synchronize(sched);
                total += ggml_time_us() - t0;
                n++;
                if (n >= 2000) break;
            }
            ret = (double) total / n;
        }
        ggml_backend_sched_free(sched);
        if (extra_cpu) {
            ggml_backend_free(extra_cpu);
        }
    }
    if (buf_a) ggml_backend_buffer_free(buf_a);
    if (buf_b) ggml_backend_buffer_free(buf_b);
    if (buf_x) ggml_backend_buffer_free(buf_x);
    ggml_free(ctx_g);
    ggml_free(ctx_x);
    ggml_free(ctx_b);
    ggml_free(ctx_a);
    return ret;
}

// per-split round trip a -> b -> a at a batch size: the mixed chain minus what its ops cost on their own devices
static double measure_split(ggml_backend_t ba, ggml_backend_t bb, int batch) {
    constexpr int n_ops = 40;
    constexpr int every = 4; // 10 excursions
    const double t_a     = bench_sched_chain(ba, bb, n_ops, 0, batch, false);
    const double t_b     = bench_sched_chain(ba, bb, n_ops, 0, batch, true);
    const double t_mixed = bench_sched_chain(ba, bb, n_ops, every, batch, false);
    if (t_a < 0 || t_b < 0 || t_mixed < 0) {
        return 0;
    }
    const int n_b = n_ops / every;
    const double own = (n_ops - n_b) * (t_a / n_ops) + n_b * (t_b / n_ops);
    return std::max(0.0, (t_mixed - own) / n_b);
}

static std::string pair_key(ggml_backend_dev_t src, ggml_backend_dev_t dst, int n_threads) {
    const bool src_cpu = ggml_backend_dev_type(src) == GGML_BACKEND_DEVICE_TYPE_CPU;
    const bool dst_cpu = ggml_backend_dev_type(dst) == GGML_BACKEND_DEVICE_TYPE_CPU;
    return fit_advisor_fingerprint(src, src_cpu ? n_threads : 0).key() + "->" + fit_advisor_fingerprint(dst, dst_cpu ? n_threads : 0).key();
}

const fit_advisor_pair_rate * fit_advisor_measurements::find_pair(ggml_backend_dev_t src, ggml_backend_dev_t dst, int n_threads) const {
    const auto it = pairs.find(pair_key(src, dst, n_threads > 0 ? n_threads : common_cpu_get_num_math()));
    return it == pairs.end() ? nullptr : &it->second;
}

void fit_advisor_measurements::ensure_pairs(const std::vector<ggml_backend_dev_t> & devs, int n_threads_in, bool force) {
    const int n_threads = n_threads_in > 0 ? n_threads_in : common_cpu_get_num_math();
    bool changed = false;

    for (ggml_backend_dev_t src : devs) {
        for (ggml_backend_dev_t dst : devs) {
            if (src == dst) {
                continue;
            }
            const std::string key = pair_key(src, dst, n_threads);
            const bool have = pairs.count(key) > 0;
            if (!force && have && pairs[key].split_n_batch_pp > 0) {
                continue;
            }
            ggml_backend_t bsrc = ggml_backend_dev_init(src, nullptr);
            ggml_backend_t bdst = ggml_backend_dev_init(dst, nullptr);
            if (!bsrc || !bdst) {
                if (bsrc) ggml_backend_free(bsrc);
                if (bdst) ggml_backend_free(bdst);
                continue;
            }
            if (ggml_backend_dev_type(src) == GGML_BACKEND_DEVICE_TYPE_CPU) ggml_backend_cpu_set_n_threads(bsrc, n_threads);
            if (ggml_backend_dev_type(dst) == GGML_BACKEND_DEVICE_TYPE_CPU) ggml_backend_cpu_set_n_threads(bdst, n_threads);
            fit_advisor_pair_rate r = have ? pairs[key] : fit_advisor_pair_rate{};
            {
                constexpr int n_batch_pp = 512;
                r.split_us_b1      = measure_split(bsrc, bdst, 1);
                r.split_us_bpp     = measure_split(bsrc, bdst, n_batch_pp);
                r.split_bytes_bpp  = (size_t) 256 * n_batch_pp * sizeof(float);
                r.split_n_batch_pp = n_batch_pp;
            }
            if (have && !force) {
                LOG_INF("%s: split %s -> %s -> %s: %.1f us at batch 1, %.1f us at batch %d\n", __func__,
                    ggml_backend_dev_name(src), ggml_backend_dev_name(dst), ggml_backend_dev_name(src), r.split_us_b1, r.split_us_bpp, r.split_n_batch_pp);
                pairs[key] = r;
                changed = true;
                ggml_backend_free(bsrc);
                ggml_backend_free(bdst);
                continue;
            }
            constexpr size_t big_bytes   = 64ull * 1024 * 1024;
            constexpr size_t small_bytes = 16 * 1024;

            for (const size_t bytes : { small_bytes, big_bytes }) {
                ggml_init_params ip = { ggml_tensor_overhead() * 4, nullptr, true };
                ggml_context * cs = ggml_init(ip);
                ggml_context * cd = ggml_init(ip);
                ggml_tensor * ts = ggml_new_tensor_1d(cs, GGML_TYPE_F32, bytes / sizeof(float));
                ggml_tensor * td = ggml_new_tensor_1d(cd, GGML_TYPE_F32, bytes / sizeof(float));
                ggml_backend_buffer_t bs = ggml_backend_alloc_ctx_tensors(cs, bsrc);
                ggml_backend_buffer_t bd = ggml_backend_alloc_ctx_tensors(cd, bdst);
                if (bs && bd) {
                    // warm up, then time
                    ggml_backend_tensor_copy_async(bsrc, bdst, ts, td);
                    ggml_backend_synchronize(bsrc);
                    ggml_backend_synchronize(bdst);
                    const int n = bytes == small_bytes ? 100 : 5;
                    const int64_t t0 = ggml_time_us();
                    for (int i = 0; i < n; i++) {
                        ggml_backend_tensor_copy_async(bsrc, bdst, ts, td);
                        ggml_backend_synchronize(bsrc);
                        ggml_backend_synchronize(bdst);
                    }
                    const double us = (double) (ggml_time_us() - t0) / n;
                    if (bytes == small_bytes) {
                        r.latency_us = us;
                    } else {
                        r.gb_s = bytes / (us * 1e-6) / 1e9;
                    }
                }
                if (bs) ggml_backend_buffer_free(bs);
                if (bd) ggml_backend_buffer_free(bd);
                ggml_free(cs);
                ggml_free(cd);
            }
            ggml_backend_free(bsrc);
            ggml_backend_free(bdst);

            LOG_INF("%s: copy %s -> %s: %.1f us small, %.2f GB/s large; split round trip %.1f us at batch 1, %.1f us at batch %d\n", __func__,
                ggml_backend_dev_name(src), ggml_backend_dev_name(dst), r.latency_us, r.gb_s, r.split_us_b1, r.split_us_bpp, r.split_n_batch_pp);
            pairs[key] = r;
            changed = true;
        }
    }
    if (changed) {
        save(default_path());
    }
}

int fit_advisor_offload_min_batch(ggml_backend_dev_t dev) {
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return 0;
    }
    // three tensors per probed batch size, thirteen sizes
    ggml_init_params ip = { ggml_tensor_overhead() * 64, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    int ret = 0;
    for (int b = 1; b <= 4096; b *= 2) {
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 4096, 4096);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, b);
        ggml_tensor * op = ggml_mul_mat(ctx, w, x);
        if (ggml_backend_dev_offload_op(dev, op)) {
            ret = b;
            break;
        }
    }
    ggml_free(ctx);
    return ret;
}
