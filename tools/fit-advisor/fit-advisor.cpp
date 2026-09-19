// llama-fit-advisor: project the memory of a set of candidate placements beside the built-in fitter's choice
//
// milestone 1: framework, inventory and probe with a fixed candidate list, no cost model yet

#include "inventory.h"
#include "probe.h"

#include "arg.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"

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

static void print_table(const std::vector<fit_advisor_candidate> & cands, fit_advisor_probe & probe) {
    constexpr double MiB = 1024.0 * 1024.0;

    printf("\n");
    printf("%-16s %8s %5s  %-34s %9s %9s %9s %9s  %-4s %9s %6s\n",
        "candidate", "n_ctx", "ngl", "device", "free", "model", "ctx+cmp", "left", "fit", "host", "t[s]");

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

        if (proj.devices.empty()) {
            printf("%-16s %8s %5d  %-34s %9s %9s %9s %9s  %-4s %9.0f %6.2f\n",
                c.name.c_str(), ctx_buf, c.n_gpu_layers, "(no devices, host only)", "", "", "", "", "-",
                proj.host.used() / MiB, proj.t_s);
            continue;
        }

        for (size_t id = 0; id < proj.devices.size(); id++) {
            const auto & d = proj.devices[id];
            const bool first = id == 0;
            printf("%-16s %8s %5d  %-34.34s %9.0f %9.0f %9.0f %9.0f  %-4s %9s %6s\n",
                first ? c.name.c_str() : "", first ? ctx_buf : "", c.n_gpu_layers,
                d.name.c_str(), d.free / MiB, d.model / MiB, (d.context + d.compute) / MiB, d.projected_free() / MiB,
                d.fits() ? "yes" : "NO",
                first ? std::to_string((long long) std::llround(proj.host.used() / MiB)).c_str() : "",
                first ? std::to_string(proj.t_s).substr(0, 5).c_str() : "");
        }
    }

    printf("\n[MiB] free: device memory free when probed; model/ctx+cmp: projected weights and context+compute buffers;\n");
    printf("left: free - projected use; fit: left >= --fit-target margin; host: projected host-side use\n");

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

int llama_fit_advisor(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FIT_PARAMS)) {
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
    return 0;
}
