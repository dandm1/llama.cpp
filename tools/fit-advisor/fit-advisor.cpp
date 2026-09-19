// llama-fit-advisor: project the memory of candidate placements beside the built-in fitter's choice,
// measure the devices, and preview the per-layer cost of each placement from the model's quantisation mix
//
// milestones so far: framework, inventory, probe, measurement; no cost-based search yet

#include "allocation.h"
#include "cost.h"
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

// candidate allocations for this milestone: a fixed set, later replaced by the search
// each is an explicit allocation turned into loader arguments by the emitter
struct named_allocation {
    std::string name;
    fit_advisor_allocation alloc;
};

static std::vector<named_allocation> build_allocations(const common_params & params, const fit_advisor_inventory & inv,
                                                       const std::vector<std::string> & device_bufts, const fit_advisor_candidate & user) {
    std::vector<named_allocation> ret;
    const uint32_t n_layer_all = inv.n_layer + inv.n_layer_nextn;
    const uint32_t ngl_max     = n_layer_all + 1; // +1 for the output layer
    const size_t   nd          = device_bufts.size();
    const uint32_t n_slots     = (uint32_t) std::max(1, params.n_parallel);

    // split a number of GPU layers across the devices in the proportion of the user's -ts, or evenly
    auto split = [&](uint32_t ngl) -> std::vector<uint32_t> {
        std::vector<uint32_t> per(nd, 0);
        if (nd == 0) {
            return per;
        }
        std::vector<float> w(nd, 1.0f);
        if (!user.tensor_split.empty()) {
            for (size_t d = 0; d < nd && d < user.tensor_split.size(); d++) {
                w[d] = user.tensor_split[d];
            }
        }
        float sum = 0;
        for (float x : w) { sum += x; }
        uint32_t assigned = 0;
        for (size_t d = 0; d < nd; d++) {
            per[d] = (uint32_t) std::lround(ngl * w[d] / sum);
            assigned += per[d];
        }
        // fix rounding on the last device
        per[nd - 1] += ngl - std::min(ngl, assigned);
        if (assigned > ngl) {
            per[nd - 1] -= std::min(per[nd - 1], assigned - ngl);
        }
        return per;
    };

    auto add = [&](const std::string & name, uint32_t ngl, uint32_t n_ctx) {
        ret.push_back({ name, fit_advisor_allocation::from_layer_split(inv, device_bufts, split(ngl), n_ctx, n_slots) });
    };

    // everything on device with the minimum context the fitter would accept
    add("ctx-min", ngl_max, (uint32_t) params.fit_params_min_ctx == UINT32_MAX ? 0 : (uint32_t) params.fit_params_min_ctx);

    // whole-layer offload at a few fractions, what -ngl gives
    for (const double frac : { 0.75, 0.5, 0.25 }) {
        const uint32_t ngl = (uint32_t) std::lround(ngl_max * frac);
        add("ngl-" + std::to_string(ngl), ngl, params.n_ctx);
    }

    // tensor-level moves: all layers on device, selected weights on the CPU
    auto move_kind = [&](const std::string & name, fit_advisor_tensor_kind kind, uint32_t il_begin, uint32_t il_end) {
        fit_advisor_allocation a = fit_advisor_allocation::from_layer_split(inv, device_bufts, split(ngl_max), params.n_ctx, n_slots);
        for (size_t i = 0; i < inv.tensors.size(); i++) {
            const auto & t = inv.tensors[i];
            if (t.kind == kind && t.layer >= (int32_t) il_begin && t.layer < (int32_t) il_end) {
                a.tensor_device[i] = fit_advisor_allocation::DEV_CPU;
            }
        }
        ret.push_back({ name, a });
    };
    if (inv.is_moe()) {
        move_kind("exps-cpu",      FIT_ADVISOR_TENSOR_FFN_EXPS, 0,                inv.n_layer);
        move_kind("exps-cpu-half", FIT_ADVISOR_TENSOR_FFN_EXPS, inv.n_layer / 2,  inv.n_layer);
    } else {
        for (const double frac : { 0.25, 0.5 }) {
            const uint32_t n_move = (uint32_t) std::lround(inv.n_layer * frac);
            move_kind("ffn-cpu-" + std::to_string(n_move), FIT_ADVISOR_TENSOR_FFN, inv.n_layer - n_move, inv.n_layer);
        }
    }
    return ret;
}

// the user's own arguments, i.e. what -fit off would load
static fit_advisor_candidate user_candidate(const common_params & params) {
    fit_advisor_candidate c;
    c.name         = "user";
    c.n_gpu_layers = params.n_gpu_layers;
    c.n_ctx        = params.n_ctx;
    c.n_slots      = (uint32_t) std::max(1, params.n_parallel);
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
    return c;
}

static std::string mib(double bytes) {
    return std::to_string((long long) std::llround(bytes / (1024.0 * 1024.0)));
}

static void print_table(const std::vector<fit_advisor_candidate> & cands, fit_advisor_probe & probe) {
    constexpr double MiB = 1024.0 * 1024.0;

    printf("\n");
    printf("%-16s %8s %3s %5s  %-34s %9s %9s %9s %8s %9s  %-4s %6s\n",
        "candidate", "n_ctx", "np", "ngl", "device", "free", "model", "ctx+cmp", "scratch", "left", "fit", "t[s]");

    for (const auto & c : cands) {
        const fit_advisor_projection & proj = probe.run(c);

        char ctx_buf[32];
        if (c.n_ctx == 0) {
            snprintf(ctx_buf, sizeof(ctx_buf), "0(%" PRIu32 ")", proj.n_ctx_train);
        } else {
            snprintf(ctx_buf, sizeof(ctx_buf), "%" PRIu32, c.n_ctx);
        }

        if (!proj.ok) {
            printf("%-16s %8s %3u %5d  probe failed: %s\n", c.name.c_str(), ctx_buf, c.n_slots, c.n_gpu_layers, proj.error.c_str());
            continue;
        }

        const std::string host_total = "total " + mib(proj.host.total);

        if (proj.devices.empty()) {
            printf("%-16s %8s %3u %5d  %-34.34s %9s %9.0f %9.0f %7.0f%s %9s  %-4s %6.2f\n",
                c.name.c_str(), ctx_buf, c.n_slots, c.n_gpu_layers, "Host (RAM)", host_total.c_str(),
                proj.host.model / MiB, (proj.host.context + proj.host.compute) / MiB,
                proj.host.scratch / MiB, proj.host.scratch_unknown ? "?" : " ", "-", "-", proj.t_s);
            continue;
        }

        for (size_t id = 0; id < proj.devices.size(); id++) {
            const auto & d = proj.devices[id];
            const bool first = id == 0;
            printf("%-16s %8s %3u %5d  %-34.34s %9.0f %9.0f %9.0f %7.0f%s %9.0f  %-4s %6s\n",
                first ? c.name.c_str() : "", first ? ctx_buf : "", c.n_slots, c.n_gpu_layers,
                d.name.c_str(), d.free / MiB, d.model / MiB, (d.context + d.compute) / MiB,
                d.scratch / MiB, d.scratch_unknown ? "?" : " ", d.projected_free() / MiB,
                d.fits() ? "yes" : "NO",
                first ? std::to_string(proj.t_s).substr(0, 5).c_str() : "");
        }
        // host: total RAM only, the CPU backend cannot report a trustworthy free figure
        printf("%-16s %8s %3s %5s  %-34.34s %9s %9.0f %9.0f %7.0f%s %9s  %-4s\n",
            "", "", "", "", "Host (RAM)", host_total.c_str(),
            proj.host.model / MiB, (proj.host.context + proj.host.compute) / MiB,
            proj.host.scratch / MiB, proj.host.scratch_unknown ? "?" : " ", "-", "-");
    }

    printf("\n[MiB] free: device memory free when probed; model/ctx+cmp: projected weights and context+compute buffers;\n");
    printf("scratch: backend pool memory the graph's ops need outside those buffers, estimated per op (? = some op had no estimate);\n");
    printf("left: free - projected use; fit: left >= --fit-target margin; Host row: projected host-side use, free RAM unknown\n");

    printf("\narguments per candidate (llama-bench takes the same flags, with ';' instead of ',' between -ot entries):\n");
    for (const auto & c : cands) {
        std::string args = "-c " + std::to_string(c.n_ctx) + " -np " + std::to_string(c.n_slots) + " -ngl " + std::to_string(c.n_gpu_layers);
        if (!c.tensor_split.empty()) {
            args += " -ts ";
            for (size_t i = 0; i < c.tensor_split.size(); i++) {
                args += (i ? "/" : "") + std::to_string((long long) std::llround(c.tensor_split[i]));
            }
        }
        if (!c.overrides.empty()) {
            args += " -ot \"" + c.overrides_str() + "\"";
        }
        printf("  %-16s %s\n", c.name.c_str(), args.c_str());
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

    fit_advisor_probe probe(params);

    // the user's own arguments come first; their probe also reveals the model's devices for the allocations
    std::vector<fit_advisor_candidate> cands = { user_candidate(params) };
    const std::vector<std::string> device_bufts = fit_advisor_device_bufts(probe.run(cands[0]));
    const std::vector<named_allocation> allocs = build_allocations(params, inv, device_bufts, cands[0]);
    for (const auto & na : allocs) {
        cands.push_back(na.alloc.to_candidate(inv, device_bufts, na.name));
    }

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

    if (params.fit_advisor_verify) {
        LOG_INF("%s: verifying that the loader honours each allocation ...\n", __func__);
        int n_bad_total = 0;
        for (const auto & na : allocs) {
            const int n_bad = fit_advisor_verify_allocation(params, inv, na.alloc, device_bufts, na.alloc.to_candidate(inv, device_bufts, na.name));
            n_bad_total += std::max(0, n_bad);
        }
        LOG_INF("%s: verification done, %d misplaced tensors in total\n", __func__, n_bad_total);
    }

    if (params.fit_advisor_no_measure) {
        return 0;
    }

    // measure the devices for the types this model actually uses
    fit_advisor_measure_options mopts;
    mopts.weight_types = inv.matmul_types(1024 * 1024); // every type used by a weight tensor of at least 1 MiB
    for (const auto & name : string_split<std::string>(params.fit_advisor_measure_types, ',')) {
        bool found = false;
        for (int t = 0; t < GGML_TYPE_COUNT && !found; t++) {
            const char * tn = ggml_type_name((ggml_type) t);
            if (tn && name == tn) {
                if (std::find(mopts.weight_types.begin(), mopts.weight_types.end(), (ggml_type) t) == mopts.weight_types.end()) {
                    mopts.weight_types.push_back((ggml_type) t);
                }
                found = true;
            }
        }
        if (!found) {
            LOG_WRN("%s: --measure-types: unknown type '%s' ignored\n", __func__, name.c_str());
        }
    }
    mopts.kv_types     = { params.cache_type_k };
    if (params.cache_type_v != params.cache_type_k) {
        mopts.kv_types.push_back(params.cache_type_v);
    }
    if (inv.head_size > 0) {
        mopts.head_size   = (int) inv.head_size;
        mopts.head_size_v = (int) inv.head_size_v;
        mopts.n_head      = (int) inv.n_head;
        mopts.n_head_kv   = (int) inv.n_head_kv;
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

    // cost of every allocation under the workload
    const fit_advisor_workload wl = [&]() {
        fit_advisor_workload w = fit_advisor_workload::preset(params.fit_advisor_workload);
        w.n_ubatch = (uint32_t) params.n_ubatch;
        w.use_mtp  = std::find(params.speculative.types.begin(), params.speculative.types.end(),
                               COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();
        return w;
    }();

    // transfers between every pair of devices, and the model's op counts per layer
    cache.ensure_pairs(devs, mopts.n_threads, params.fit_advisor_remeasure);

    const fit_advisor_graph_profile gp = probe.graph_profile(inv.n_layer + inv.n_layer_nextn);
    if (gp.ok) {
        uint32_t sum_tg = 0, sum_pp = 0;
        for (uint32_t x : gp.ops_per_layer_tg) sum_tg += x;
        for (uint32_t x : gp.ops_per_layer_pp) sum_pp += x;
        LOG_INF("%s: graph ops: %u nodes at batch 1 (%.1f per layer, %u global), %u nodes at batch %u (%.1f per layer, %u global)\n", __func__,
            gp.n_nodes_tg, gp.ops_per_layer_tg.empty() ? 0.0 : (double) sum_tg / gp.ops_per_layer_tg.size(), gp.ops_global_tg,
            gp.n_nodes_pp, gp.n_batch_pp, gp.ops_per_layer_pp.empty() ? 0.0 : (double) sum_pp / gp.ops_per_layer_pp.size(), gp.ops_global_pp);
    } else {
        LOG_WRN("%s: graph profile unavailable: %s\n", __func__, gp.error.c_str());
    }

    std::vector<fit_advisor_cost_device> cost_devs;
    std::vector<ggml_backend_dev_t>      cost_dev_handles;
    for (const auto & buft : device_bufts) {
        fit_advisor_cost_device cd;
        cd.name   = buft;
        cd.n_embd = inv.n_embd;
        ggml_backend_dev_t handle = nullptr;
        for (size_t i = 0; i < devs.size(); i++) {
            if (buft == ggml_backend_dev_name(devs[i])) {
                cd.meas = meas[i];
                cd.offload_min_batch = fit_advisor_offload_min_batch(devs[i]);
                handle = devs[i];
            }
        }
        cost_devs.push_back(cd);
        cost_dev_handles.push_back(handle);
    }
    {
        fit_advisor_cost_device cpu;
        cpu.name   = "CPU";
        cpu.is_cpu = true;
        cpu.n_embd = inv.n_embd;
        ggml_backend_dev_t handle = nullptr;
        for (size_t i = 0; i < devs.size(); i++) {
            if (ggml_backend_dev_type(devs[i]) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                cpu.meas = meas[i];
                handle = devs[i];
            }
        }
        cost_devs.push_back(cpu);
        cost_dev_handles.push_back(handle);
    }
    fit_advisor_pair_table pair_table(cost_devs.size(), std::vector<fit_advisor_pair_rate>(cost_devs.size()));
    for (size_t a = 0; a < cost_devs.size(); a++) {
        for (size_t b = 0; b < cost_devs.size(); b++) {
            if (a != b && cost_dev_handles[a] && cost_dev_handles[b]) {
                if (const auto * r = cache.find_pair(cost_dev_handles[a], cost_dev_handles[b], mopts.n_threads)) {
                    pair_table[a][b] = *r;
                }
            }
        }
    }

    printf("\nestimated cost per request, workload '%s': %u prompt + %u generated tokens, %u concurrent, ubatch %u\n",
        params.fit_advisor_workload.c_str(), wl.prompt_tokens, wl.gen_tokens, wl.concurrency, wl.n_ubatch);
    for (const auto & cd : cost_devs) {
        if (!cd.is_cpu) {
            printf("  %s runs CPU-resident weights itself from batch %d\n", cd.name.c_str(), cd.offload_min_batch);
        }
    }
    printf("%-16s %4s %9s %9s %9s  %9s %9s %9s %9s\n",
        "candidate", "fit", "gen tok/s", "pp tok/s", "request s", "weights", "attn", "overhead", "boundary");
    printf("%-16s %4s %9s %9s %9s  %9s %9s %9s %9s\n", "", "", "", "", "", "[us/step]", "", "", "");
    for (const auto & na : allocs) {
        const fit_advisor_candidate  cand = na.alloc.to_candidate(inv, device_bufts, na.name);
        const fit_advisor_projection & pj = probe.run(cand);
        const fit_advisor_cost cost = fit_advisor_cost_estimate(inv, na.alloc, pj, gp, cost_devs, pair_table, wl);
        if (!cost.ok) {
            printf("%-16s %4s cost unavailable: %s\n", na.name.c_str(), pj.fits_all() ? "yes" : "NO", cost.error.c_str());
            continue;
        }
        printf("%-16s %4s %9.1f %9.0f %9.2f  %9.0f %9.0f %9.0f %9.0f%s\n",
            na.name.c_str(), pj.fits_all() ? "yes" : "NO", cost.gen_tokens_per_s, cost.prompt_tokens_per_s, cost.t_request_us * 1e-6,
            cost.step_weights_us, cost.step_attn_us, cost.step_overhead_us, cost.step_boundary_us,
            cost.error.empty() ? "" : ("  (" + cost.error + ")").c_str());
    }
    return 0;
}
