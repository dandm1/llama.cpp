// llama-fit-advisor: project the memory of candidate placements beside the built-in fitter's choice,
// measure the devices, and preview the per-layer cost of each placement from the model's quantisation mix
//
// milestones so far: framework, inventory, probe, measurement; no cost-based search yet

#include "inventory.h"
#include "measure.h"
#include "probe.h"

#include "arg.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// satisfies -Wmissing-declarations
int llama_fit_advisor(int argc, char ** argv);

// regex alternation matching the layer indices [il_begin, il_end)
static std::string layer_range_pattern(uint32_t il_begin, uint32_t il_end) {
    std::string ret = "(";
    for (uint32_t il = il_begin; il < il_end; il++) {
        ret += (il > il_begin ? "|" : "") + std::to_string(il);
    }
    return ret + ")";
}

// the fixed candidate set for this milestone
static std::vector<fit_advisor_candidate> build_candidates(const common_params & params, const fit_advisor_inventory & inv) {
    std::vector<fit_advisor_candidate> ret;

    const uint32_t n_layer_all = inv.n_layer + inv.n_layer_nextn;
    const uint32_t ngl_max     = n_layer_all + 1; // +1 for the output layer

    // exactly what the user asked for, i.e. what -fit off would load
    {
        fit_advisor_candidate c;
        c.name         = "user";
        c.n_gpu_layers = params.n_gpu_layers;
        c.n_ctx        = params.n_ctx;
        for (size_t i = 0; i < llama_max_devices(); i++) {
            if (params.tensor_split[i] != 0.0f) {
                c.tensor_split.assign(params.tensor_split, params.tensor_split + i + 1);
            }
        }
        for (const auto & o : params.tensor_buft_overrides) {
            if (o.pattern) {
                c.overrides.push_back({ o.pattern, ggml_backend_buft_name(o.buft) });
            }
        }
        ret.push_back(c);
    }

    // everything on device with the minimum context the fitter would accept
    {
        fit_advisor_candidate c;
        c.name         = "ctx-min";
        c.n_gpu_layers = -1;
        c.n_ctx        = (uint32_t) params.fit_params_min_ctx == UINT32_MAX ? 0 : (uint32_t) params.fit_params_min_ctx;
        ret.push_back(c);
    }

    // whole-layer offload at a few fractions, what -ngl gives
    for (const double frac : { 0.75, 0.5, 0.25 }) {
        fit_advisor_candidate c;
        c.n_gpu_layers = (int32_t) std::lround(ngl_max * frac);
        c.n_ctx        = params.n_ctx;
        c.name         = "ngl-" + std::to_string(c.n_gpu_layers);
        ret.push_back(c);
    }

    if (inv.is_moe()) {
        // all experts in host memory, dense parts on device
        fit_advisor_candidate c;
        c.name         = "exps-cpu";
        c.n_gpu_layers = -1;
        c.n_ctx        = params.n_ctx;
        c.overrides.push_back({ R"(blk\.\d+\.ffn_(up|down|gate|gate_up)_(ch|)exps)", "CPU" });
        ret.push_back(c);

        // experts of the second half of the layers in host memory
        fit_advisor_candidate h;
        h.name         = "exps-cpu-half";
        h.n_gpu_layers = -1;
        h.n_ctx        = params.n_ctx;
        h.overrides.push_back({ R"(blk\.)" + layer_range_pattern(inv.n_layer / 2, inv.n_layer) + R"(\.ffn_(up|down|gate|gate_up)_(ch|)exps)", "CPU" });
        ret.push_back(h);
    } else {
        // dense FFN tensors of the last quarter / half of the layers in host memory, KV cache stays on device
        for (const double frac : { 0.25, 0.5 }) {
            const uint32_t n_move = (uint32_t) std::lround(inv.n_layer * frac);
            fit_advisor_candidate c;
            c.name         = "ffn-cpu-" + std::to_string(n_move);
            c.n_gpu_layers = -1;
            c.n_ctx        = params.n_ctx;
            c.overrides.push_back({ R"(blk\.)" + layer_range_pattern(inv.n_layer - n_move, inv.n_layer) + R"(\.ffn_(gate|up|down|gate_up)\.weight)", "CPU" });
            ret.push_back(c);
        }
    }

    return ret;
}

static std::string mib(double bytes) {
    return std::to_string((long long) std::llround(bytes / (1024.0 * 1024.0)));
}

static void print_table(const std::vector<fit_advisor_candidate> & cands, fit_advisor_probe & probe) {
    constexpr double MiB = 1024.0 * 1024.0;

    printf("\n");
    printf("%-16s %8s %5s  %-34s %9s %9s %9s %9s  %-4s %6s\n",
        "candidate", "n_ctx", "ngl", "device", "free", "model", "ctx+cmp", "left", "fit", "t[s]");

    for (const auto & c : cands) {
        const fit_advisor_projection & proj = probe.run(c);

        char ctx_buf[32];
        if (c.n_ctx == 0) {
            snprintf(ctx_buf, sizeof(ctx_buf), "0(%" PRIu32 ")", proj.n_ctx_train);
        } else {
            snprintf(ctx_buf, sizeof(ctx_buf), "%" PRIu32, c.n_ctx);
        }

        if (!proj.ok) {
            printf("%-16s %8s %5d  probe failed: %s\n", c.name.c_str(), ctx_buf, c.n_gpu_layers, proj.error.c_str());
            continue;
        }

        const std::string host_total = "total " + mib(proj.host.total);

        if (proj.devices.empty()) {
            printf("%-16s %8s %5d  %-34.34s %9s %9.0f %9.0f %9s  %-4s %6.2f\n",
                c.name.c_str(), ctx_buf, c.n_gpu_layers, "Host (RAM)", host_total.c_str(),
                proj.host.model / MiB, (proj.host.context + proj.host.compute) / MiB, "-", "-", proj.t_s);
            continue;
        }

        for (size_t id = 0; id < proj.devices.size(); id++) {
            const auto & d = proj.devices[id];
            const bool first = id == 0;
            printf("%-16s %8s %5d  %-34.34s %9.0f %9.0f %9.0f %9.0f  %-4s %6s\n",
                first ? c.name.c_str() : "", first ? ctx_buf : "", c.n_gpu_layers,
                d.name.c_str(), d.free / MiB, d.model / MiB, (d.context + d.compute) / MiB, d.projected_free() / MiB,
                d.fits() ? "yes" : "NO",
                first ? std::to_string(proj.t_s).substr(0, 5).c_str() : "");
        }
        // host: total RAM only, the CPU backend cannot report a trustworthy free figure
        printf("%-16s %8s %5s  %-34.34s %9s %9.0f %9.0f %9s  %-4s\n",
            "", "", "", "Host (RAM)", host_total.c_str(),
            proj.host.model / MiB, (proj.host.context + proj.host.compute) / MiB, "-", "-");
    }

    printf("\n[MiB] free: device memory free when probed; model/ctx+cmp: projected weights and context+compute buffers;\n");
    printf("left: free - projected use; fit: left >= --fit-target margin; Host row: projected host-side use, free RAM unknown\n");

    bool any_overrides = false;
    for (const auto & c : cands) {
        if (!c.overrides.empty()) {
            if (!any_overrides) {
                printf("\ntensor overrides (-ot) per candidate:\n");
                any_overrides = true;
            }
            printf("  %-16s %s\n", c.name.c_str(), c.overrides_str().c_str());
        }
    }
}

// devices to measure: the model's devices plus the CPU, which is always a placement target
static std::vector<ggml_backend_dev_t> devices_to_measure(const fit_advisor_projection & proj) {
    std::vector<ggml_backend_dev_t> ret;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const std::string name = ggml_backend_dev_name(dev);
        bool used = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
        for (const auto & d : proj.devices) {
            if (d.name.rfind(name + " (", 0) == 0) {
                used = true;
            }
        }
        if (used) {
            ret.push_back(dev);
        }
    }
    return ret;
}

// per-layer generation cost preview: bytes of each tensor divided by the device's measured rate for that tensor's type
// experts count only for the fraction read per token; tensors with no measured rate for their type are reported
static void print_layer_costs(const fit_advisor_inventory & inv, const std::vector<ggml_backend_dev_t> & devs,
                              const std::vector<const fit_advisor_device_measurements *> & meas) {
    constexpr double MiB = 1024.0 * 1024.0;

    printf("\nper-layer generation cost preview [us per token, weights only, no attention over the KV cache]\n");
    printf("%-6s %8s  %-40s", "layer", "MiB", "type mix");
    for (const auto & dev : devs) {
        printf(" %12.12s", ggml_backend_dev_name(dev));
    }
    printf("\n");

    std::map<ggml_type, size_t> unmeasured; // type -> bytes, across the whole model

    const uint32_t n_all = (uint32_t) inv.layers.size();
    for (uint32_t il = 0; il < n_all; il++) {
        // effective bytes per type in this layer, experts scaled by the active fraction
        std::map<ggml_type, double> bytes_by_type;
        double bytes_total = 0;
        for (const auto & t : inv.tensors) {
            if (t.layer != (int32_t) il) {
                continue;
            }
            const double b = t.kind == FIT_ADVISOR_TENSOR_FFN_EXPS ? t.nbytes * inv.expert_active_fraction() : (double) t.nbytes;
            bytes_by_type[t.type] += b;
            bytes_total += b;
        }

        std::vector<std::pair<ggml_type, double>> mix(bytes_by_type.begin(), bytes_by_type.end());
        std::sort(mix.begin(), mix.end(), [](const auto & a, const auto & b) { return a.second > b.second; });
        std::string mix_str;
        for (size_t i = 0; i < mix.size() && i < 3; i++) {
            if (mix[i].second < 0.01 * bytes_total) {
                break; // norms, biases and the like
            }
            char buf[48];
            snprintf(buf, sizeof(buf), "%s%s %.0f%%", i ? " " : "", ggml_type_name(mix[i].first), 100.0 * mix[i].second / bytes_total);
            mix_str += buf;
        }

        printf("%-6" PRIu32 " %8.0f  %-40.40s", il, bytes_total / MiB, mix_str.c_str());
        for (size_t d = 0; d < devs.size(); d++) {
            double us = 0;
            bool complete = true;
            for (const auto & [type, b] : bytes_by_type) {
                const auto it = meas[d]->matmul.find(ggml_type_name(type));
                if (it == meas[d]->matmul.end() || !it->second.supported || it->second.bytes_per_s <= 0) {
                    // small unmeasured tensors are norms and biases, not matmuls; only flag a real share of the layer
                    if (b >= 0.02 * bytes_total) {
                        complete = false;
                        unmeasured[type] += (size_t) b;
                    }
                    continue;
                }
                us += b / it->second.bytes_per_s * 1e6;
            }
            printf(" %11.0f%s", us, complete ? " " : "?");
        }
        printf("%s\n", il >= inv.n_layer ? "  (MTP)" : "");
    }

    if (!unmeasured.empty()) {
        printf("? = incomplete, a significant share of the layer has no measured rate for: ");
        for (const auto & [type, b] : unmeasured) {
            printf("%s ", ggml_type_name(type));
        }
        printf("\n");
    }
}

int llama_fit_advisor(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FIT_ADVISOR)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    fit_advisor_inventory inv;
    try {
        inv = fit_advisor_inventory_load(params.model.path);
    } catch (const std::exception & e) {
        LOG_ERR("%s: %s\n", __func__, e.what());
        return 1;
    }
    fit_advisor_inventory_print(inv);

    std::vector<fit_advisor_candidate> cands = build_candidates(params, inv);

    fit_advisor_probe probe(params);

    // the built-in fitter's answer for the same arguments, for comparison
    {
        fit_advisor_candidate fit;
        const common_params_fit_status status = probe.fitter_choice(fit);
        switch (status) {
            case COMMON_PARAMS_FIT_STATUS_SUCCESS:
                LOG_INF("%s: built-in fitter succeeded: %s\n", __func__, fit.key().c_str());
                break;
            case COMMON_PARAMS_FIT_STATUS_FAILURE:
                LOG_WRN("%s: built-in fitter could not fit, its partial result: %s\n", __func__, fit.key().c_str());
                fit.name = "fit(failed)";
                break;
            case COMMON_PARAMS_FIT_STATUS_ERROR:
                LOG_WRN("%s: built-in fitter hit an error, its partial result: %s\n", __func__, fit.key().c_str());
                fit.name = "fit(error)";
                break;
        }
        cands.insert(cands.begin() + 1, fit);
    }

    LOG_INF("%s: probing %zu candidates ...\n", __func__, cands.size());
    common_log_flush(common_log_main());

    print_table(cands, probe);
    LOG_INF("%s: %zu probes executed\n", __func__, probe.n_probes);

    if (params.fit_advisor_no_measure) {
        return 0;
    }

    // measure the devices for the types this model actually uses
    fit_advisor_measure_options mopts;
    mopts.weight_types = inv.weight_types(0.01); // every type holding at least 1% of the bytes
    mopts.kv_types     = { params.cache_type_k };
    if (params.cache_type_v != params.cache_type_k) {
        mopts.kv_types.push_back(params.cache_type_v);
    }
    if (inv.head_size > 0) {
        mopts.head_size = (int) inv.head_size;
        mopts.n_head    = (int) inv.n_head;
        mopts.n_head_kv = (int) inv.n_head_kv;
    }
    mopts.n_batch_pp = params.n_ubatch;
    mopts.n_threads  = params.cpuparams.n_threads;
    mopts.verbose    = params.verbosity >= LOG_LEVEL_DEBUG;

    {
        std::string types;
        for (const ggml_type t : mopts.weight_types) {
            types += std::string(types.empty() ? "" : ",") + ggml_type_name(t);
        }
        LOG_INF("%s: measuring devices for weight types %s, KV %s, head size %d (cached results are reused)\n", __func__,
            types.c_str(), ggml_type_name(mopts.kv_types[0]), mopts.head_size);
    }

    fit_advisor_measurements cache;
    cache.load(fit_advisor_measurements::default_path());

    const fit_advisor_projection & proj0 = probe.run(cands[0]);
    const std::vector<ggml_backend_dev_t> devs = devices_to_measure(proj0);
    std::vector<const fit_advisor_device_measurements *> meas;
    for (const auto & dev : devs) {
        meas.push_back(&cache.ensure(dev, mopts, params.fit_advisor_remeasure));
    }
    for (const auto * m : meas) {
        fit_advisor_measurements_print(*m);
    }
    common_log_flush(common_log_main());

    print_layer_costs(inv, devs, meas);
    return 0;
}
