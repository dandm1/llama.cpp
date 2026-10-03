// llama-fit-advisor: project the memory of candidate placements beside the built-in fitter's choice,
// measure the devices, and preview the per-layer cost of each placement from the model's quantisation mix
//
// milestones so far: framework, inventory, probe, measurement; no cost-based search yet

#include "allocation.h"
#include "cost.h"
#include "emit.h"
#include "inventory.h"
#include "measure.h"
#include "probe.h"
#include "search.h"
#include "validate.h"

#include "arg.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// satisfies -Wmissing-declarations
int llama_fit_advisor(int argc, char ** argv);

// the user's op offload thresholds as a per-device list (empty when everything is at the default)
static std::vector<int32_t> user_offload_dev(const common_params & params) {
    std::vector<int32_t> ret = params.op_offload_min_batch_dev;
    if (ret.empty() && params.op_offload_min_batch > 0) {
        ret.assign(llama_max_devices(), params.op_offload_min_batch);
    }
    return ret;
}

// candidate allocations for this milestone: a fixed set, later replaced by the search
// each is an explicit allocation turned into loader arguments by the emitter
struct named_allocation {
    std::string name;
    fit_advisor_allocation alloc;
};

// the allocation a whole-layer candidate (-ngl, -ts) describes, as the loader would place it: a leading block on the
// CPU, then one block per device in the proportion of the split. the fitter's choice is such a candidate
static fit_advisor_allocation allocation_from_candidate(const fit_advisor_inventory & inv, const std::vector<std::string> & device_bufts,
                                                        const fit_advisor_candidate & cand, uint32_t n_slots) {
    const uint32_t n_layer_all = inv.n_layer + inv.n_layer_nextn;
    const uint32_t ngl_max     = n_layer_all + 1;
    const size_t   nd          = device_bufts.size();
    const uint32_t ngl         = cand.n_gpu_layers < 0 ? ngl_max : std::min<uint32_t>(ngl_max, (uint32_t) cand.n_gpu_layers);
    std::vector<float> w(nd, 1.0f);
    for (size_t d = 0; d < nd && d < cand.tensor_split.size(); d++) {
        w[d] = cand.tensor_split[d];
    }
    float sum = 0;
    for (float x : w) { sum += x; }
    std::vector<uint32_t> per(nd, 0);
    uint32_t assigned = 0;
    for (size_t d = 0; d < nd; d++) {
        per[d] = sum > 0 ? (uint32_t) std::lround(ngl * w[d] / sum) : 0;
        assigned += per[d];
    }
    if (nd > 0) {
        per[nd - 1] += ngl - std::min(ngl, assigned);
        if (assigned > ngl) {
            per[nd - 1] -= std::min(per[nd - 1], assigned - ngl);
        }
    }
    fit_advisor_allocation a = fit_advisor_allocation::from_layer_split(inv, device_bufts, per, cand.n_ctx, n_slots);
    a.n_ubatch      = cand.n_ubatch;
    a.draft_mtp     = cand.spec_mtp;
    a.mtp_draft_n   = cand.spec_draft_n;
    a.flash_attn    = cand.flash_attn;
    a.no_kv_offload = cand.no_kv_offload;
    a.op_offload_min_batch_dev = cand.op_offload_min_batch_dev;
    return a;
}

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

    const bool spec_mtp = std::find(params.speculative.types.begin(), params.speculative.types.end(),
                                    COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();
    auto add = [&](const std::string & name, uint32_t ngl, uint32_t n_ctx) {
        fit_advisor_allocation a = fit_advisor_allocation::from_layer_split(inv, device_bufts, split(ngl), n_ctx, n_slots);
        a.draft_mtp = spec_mtp;
        a.op_offload_min_batch_dev = user_offload_dev(params);
        a.flash_attn    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO ? -1 : params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_ENABLED ? 1 : 0;
        a.no_kv_offload = params.no_kv_offload;
        ret.push_back({ name, a });
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
        a.draft_mtp = spec_mtp;
        a.op_offload_min_batch_dev = user_offload_dev(params);
        a.flash_attn    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO ? -1 : params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_ENABLED ? 1 : 0;
        a.no_kv_offload = params.no_kv_offload;
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
    c.spec_mtp = std::find(params.speculative.types.begin(), params.speculative.types.end(),
                           COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();
    c.op_offload_min_batch_dev = user_offload_dev(params);
    c.flash_attn    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO ? -1 : params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_ENABLED ? 1 : 0;
    c.no_kv_offload = params.no_kv_offload;
    // -old as given, translated to buffer type names like the rest of the candidate
    for (const auto & o : params.layer_dev_overrides) {
        if (o.il == -1) {
            break;
        }
        ggml_backend_buffer_type_t buft = o.dev ? ggml_backend_dev_buffer_type(o.dev) : nullptr;
        c.layer_devices += (c.layer_devices.empty() ? "" : ",") + (o.il == LLAMA_LAYER_OUTPUT ? std::string("output") : std::to_string(o.il))
                         + "=" + (buft ? ggml_backend_buft_name(buft) : "CPU");
    }
    return c;
}

static std::string mib(double bytes) {
    return std::to_string((long long) std::llround(bytes / (1024.0 * 1024.0)));
}

static void print_table(const std::vector<fit_advisor_candidate> & cands, fit_advisor_probe & probe, int32_t n_batch, const common_params & params) {
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
    printf("left: free - projected use; fit: left >= margin (--fit-target; default 512 MiB where scratch is estimated, else 1024,\n");
    printf("  and the measured runtime overhead + %.0f MiB once the devices have been measured or validated);\n", FIT_ADVISOR_MARGIN_PAD / MiB);
    printf("Host row: projected host-side use, free RAM unknown\n");

    printf("\narguments per candidate (llama-bench takes the same flags, with ';' instead of ',' between -ot entries):\n");
    for (const auto & c : cands) {
        std::string args = "-c " + std::to_string(c.n_ctx) + " -np " + std::to_string(c.n_slots) + " -ngl " + std::to_string(c.n_gpu_layers);
        if (c.n_ubatch > 0) {
            args += " -ub " + std::to_string(c.n_ubatch) + " -b " + std::to_string(std::max<uint32_t>(c.n_ubatch, (uint32_t) n_batch));
        }
        if (!c.tensor_split.empty()) {
            args += " -ts ";
            for (size_t i = 0; i < c.tensor_split.size(); i++) {
                args += (i ? "/" : "") + std::to_string((long long) std::llround(c.tensor_split[i]));
            }
        }
        if (!c.overrides.empty()) {
            args += " -ot \"" + c.overrides_str() + "\"";
        }
        if (!c.layer_devices.empty()) {
            args += " -old " + c.layer_devices_cli();
        }
        args += fit_advisor_passthrough_cli(params, c);
        printf("  %-16s %s\n", c.name.c_str(), args.c_str());
    }
    fflush(stdout); // stdout is block-buffered when redirected, keep the tables separate from the log lines
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
    fflush(stdout);
}

static void emit_if_requested(const common_params & params, const fit_advisor_candidate & cand, const char * what) {
    if (params.fit_advisor_emit_ini.empty()) {
        return;
    }
    std::string name = params.fit_advisor_emit_name;
    if (name.empty()) {
        name = params.model.path;
        const size_t slash = name.find_last_of("/\\");
        if (slash != std::string::npos) {
            name = name.substr(slash + 1);
        }
        const size_t dot = name.rfind(".gguf");
        if (dot != std::string::npos) {
            name = name.substr(0, dot);
        }
    }
    const fit_advisor_emit_result er = fit_advisor_emit_ini(params.fit_advisor_emit_ini, name, params.model.path, cand, params.n_batch,
                                                            fit_advisor_passthrough(params, cand));
    if (er.ok) {
        LOG_INF("%s: wrote the %s allocation as section [%s] to %s and verified it reloads unchanged\n", __func__, what, er.section.c_str(), er.path.c_str());
        common_log_flush(common_log_main());
        printf("\n%s", er.ini.c_str());
        printf("use it with: llama-server --models-preset %s   (router mode; or copy the keys into ~/.config/llama.cpp/config.ini)\n", er.path.c_str());
        fflush(stdout);
    } else {
        LOG_ERR("%s: emitting the preset failed: %s\n", __func__, er.error.c_str());
    }
}

// run the search and print the result: request cost, per-device memory, what sits where, and the loader arguments
static fit_advisor_search_result search_and_report(const common_params & params, const fit_advisor_inventory & inv, fit_advisor_probe & probe,
                                                   const std::vector<std::string> & device_bufts, const fit_advisor_graph_profile & gp,
                                                   const std::vector<fit_advisor_cost_device> & cost_devs, const fit_advisor_pair_table & pair_table,
                                                   const fit_advisor_workload & wl, const fit_advisor_search_options & sopts) {
    LOG_INF("%s: searching allocations (%d annealing iterations) ...\n", __func__, sopts.anneal_iters);
    const int64_t t_search0 = ggml_time_us();
    const fit_advisor_search_result sr = fit_advisor_search(inv, probe, device_bufts, gp, cost_devs, pair_table, wl, sopts);
    const double t_search = (ggml_time_us() - t_search0) * 1e-6;
    if (!sr.ok) {
        LOG_WRN("%s: search found no feasible allocation\n", __func__);
        return sr;
    }
    LOG_INF("%s: search done in %.1f s: %d seed cells, %d probes, %d annealing moves accepted\n", __func__,
        t_search, sr.n_cells, sr.n_probes, sr.n_anneal_accepted);
    common_log_flush(common_log_main());

    printf("\nbest allocation: %s\n", sr.name.c_str());
    printf("  request %.2f s (best seed %.2f s), gen %.2f tok/s, pp %.0f tok/s, ubatch %u, slots %u\n",
        sr.cost.t_request_us * 1e-6, sr.seed_request_us * 1e-6, sr.cost.gen_tokens_per_s, sr.cost.prompt_tokens_per_s,
        sr.wl.n_ubatch, sr.alloc.n_slots);
    printf("  generation step %.0f us: weights %.0f, attention %.0f, per-node overhead %.0f, boundaries %.0f\n",
        sr.cost.t_gen_step_us, sr.cost.step_weights_us, sr.cost.step_attn_us, sr.cost.step_overhead_us, sr.cost.step_boundary_us);
    if (sr.alloc.flash_attn >= 0 || sr.alloc.no_kv_offload) {
        printf("  attention: flash attention %s%s\n", sr.alloc.flash_attn < 0 ? "auto" : sr.alloc.flash_attn ? "on" : "off",
            sr.alloc.no_kv_offload ? ", KV cache in host memory (-nkvo)" : "");
    }
    for (size_t d = 0; d < sr.alloc.op_offload_min_batch_dev.size() && d < device_bufts.size(); d++) {
        const int32_t v = sr.alloc.op_offload_min_batch_dev[d];
        if (v > 0) {
            printf("  %s takes CPU-resident weights from batch %s (the device default is %d)\n", device_bufts[d].c_str(),
                v >= FIT_ADVISOR_OFFLOAD_NEVER ? "never: its layers' offloads go to the first willing device" : std::to_string(v).c_str(),
                fit_advisor_default_op_offload());
        }
    }
    auto print_by_depth = [&]() {
        printf("    gen tok/s by draft depth:");
        for (size_t d = 0; d < sr.cost.mtp_depth_tok_s.size(); d++) {
            printf(" %zu: %.2f", d, sr.cost.mtp_depth_tok_s[d]);
        }
        printf("  (the scan stops at the first depth that is worse)\n");
    };
    if (sr.wl.use_mtp) {
        printf("  drafting ON at depth %u: verification batch of %u, MTP draft run %.0f us x %u, %.2f tokens per step\n",
            sr.cost.mtp_draft_n, 1 + sr.cost.mtp_draft_n, sr.cost.t_mtp_draft_us, sr.cost.mtp_draft_n, sr.cost.tokens_per_step);
        print_by_depth();
    } else if (inv.n_layer_nextn > 0 && params.fit_advisor_mtp) {
        printf("  drafting OFF: the MTP layers are not loaded; drafting was priced on every allocation and did not pay\n");
        if (!sr.cost.mtp_depth_tok_s.empty()) {
            print_by_depth();
        }
    }
    for (size_t d = 0; d < sr.proj.devices.size(); d++) {
        const auto & pd = sr.proj.devices[d];
        printf("  %-34.34s model %6.0f MiB, ctx+cmp %5.0f, scratch %4.0f, left %6.0f MiB%s\n", pd.name.c_str(),
            pd.model / (1024.0 * 1024), (pd.context + pd.compute) / (1024.0 * 1024), pd.scratch / (1024.0 * 1024),
            pd.projected_free() / (1024.0 * 1024), pd.fits() ? "" : "  (over budget!)");
    }
    // what sits on each device, by tensor kind and layer ranges
    const uint32_t n_layer_all = inv.n_layer + inv.n_layer_nextn;
    for (size_t d = 0; d < device_bufts.size(); d++) {
        std::map<fit_advisor_tensor_kind, std::vector<uint32_t>> layers_by_kind;
        size_t bytes_on = 0;
        for (size_t i = 0; i < inv.tensors.size(); i++) {
            if (sr.alloc.tensor_device[i] != (int) d) continue;
            if (inv.tensors[i].layer >= (int32_t) inv.n_layer && !sr.alloc.draft_mtp) continue; // MTP layer not loaded
            bytes_on += inv.tensors[i].nbytes;
            if (inv.tensors[i].layer >= 0) layers_by_kind[inv.tensors[i].kind].push_back((uint32_t) inv.tensors[i].layer);
        }
        printf("  %s holds %.0f MiB of weights, layers %s\n", device_bufts[d].c_str(), bytes_on / (1024.0 * 1024),
            [&]() { // layer range of this device's block
                std::string r;
                uint32_t first = UINT32_MAX, last = 0;
                for (uint32_t il = 0; il <= n_layer_all; il++) {
                    if (sr.alloc.layer_device(il, n_layer_all) == (int) d) { first = std::min(first, il); last = il; }
                }
                return first == UINT32_MAX ? std::string("none") : std::to_string(first) + "-" + std::to_string(last);
            }().c_str());
        for (auto & [kind, layers] : layers_by_kind) {
            std::sort(layers.begin(), layers.end());
            layers.erase(std::unique(layers.begin(), layers.end()), layers.end());
            // compress into ranges
            std::string ranges;
            for (size_t k = 0; k < layers.size();) {
                size_t j = k;
                while (j + 1 < layers.size() && layers[j + 1] == layers[j] + 1) j++;
                ranges += (ranges.empty() ? "" : ",") + std::to_string(layers[k]) + (j > k ? "-" + std::to_string(layers[j]) : "");
                k = j + 1;
            }
            printf("    %-12s layers %s\n", fit_advisor_tensor_kind_name(kind), ranges.c_str());
        }
    }
    for (const auto & [il, dev] : sr.alloc.layer_home) {
        printf("  layer %s re-homed to %s (-old): its KV cache and state go with it\n",
            il == n_layer_all ? "output" : std::to_string(il).c_str(), dev < 0 ? "CPU" : device_bufts[dev].c_str());
    }
    {
        const fit_advisor_candidate & c = sr.cand;
        std::string args = "-c " + std::to_string(c.n_ctx) + " -np " + std::to_string(c.n_slots) + " -ngl " + std::to_string(c.n_gpu_layers);
        if (c.n_ubatch > 0) {
            args += " -ub " + std::to_string(c.n_ubatch) + " -b " + std::to_string(std::max<uint32_t>(c.n_ubatch, (uint32_t) params.n_batch));
        }
        if (!c.tensor_split.empty()) {
            args += " -ts ";
            for (size_t i = 0; i < c.tensor_split.size(); i++) args += (i ? "/" : "") + std::to_string((long long) std::llround(c.tensor_split[i]));
        }
        if (!c.overrides.empty()) args += " -ot \"" + c.overrides_str() + "\"";
        if (!c.layer_devices.empty()) args += " -old " + c.layer_devices_cli();
        args += fit_advisor_passthrough_cli(params, c);
        printf("  args: %s\n", args.c_str());
    }
    fflush(stdout);
    return sr;
}

int llama_fit_advisor(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FIT_ADVISOR)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // the op offload threshold is a device setting; apply the user's before anything reads it, and remember what
    // the devices had so candidates without an explicit value can restore it
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            const int t = fit_advisor_offload_min_batch(ggml_backend_dev_get(i));
            if (t > 0) {
                fit_advisor_set_default_op_offload(t);
                break;
            }
        }
    }

    // CPU-resident weights go to a pinned host buffer type when asked; the name comes from the first device that has one
    std::string cpu_buft = "CPU";
    if (params.fit_advisor_pin_cpu_weights) {
        for (size_t i = 0; i < ggml_backend_dev_count() && cpu_buft == "CPU"; i++) {
            if (ggml_backend_buffer_type_t hb = ggml_backend_dev_host_buffer_type(ggml_backend_dev_get(i))) {
                cpu_buft = ggml_backend_buft_name(hb);
            }
        }
        if (cpu_buft == "CPU") {
            LOG_WRN("%s: --pin-cpu-weights: no device offers a pinned host buffer type, using CPU\n", __func__);
        } else {
            LOG_INF("%s: CPU-resident weights will be placed in %s (pinned host memory)\n", __func__, cpu_buft.c_str());
        }
    }
    fit_advisor_allocation::default_cpu_buft = cpu_buft;


    fit_advisor_inventory inv;
    try {
        inv = fit_advisor_inventory_load(params.model.path);
    } catch (const std::exception & e) {
        LOG_ERR("%s: %s\n", __func__, e.what());
        return 1;
    }
    fit_advisor_inventory_print(inv);

    // MTP drafting: considered whenever the model has MTP layers and --no-mtp was not given. downstream (probe, cost,
    // search, emit) the presence of draft-mtp in the speculative types means "drafting may be chosen"
    params.speculative.types.erase(std::remove(params.speculative.types.begin(), params.speculative.types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP),
                                   params.speculative.types.end());
    if (inv.n_layer_nextn == 0) {
        LOG_INF("%s: the model has no MTP layers, drafting is not considered\n", __func__);
    } else if (!params.fit_advisor_mtp) {
        LOG_INF("%s: --no-mtp: the model's %u MTP layer(s) are not loaded and drafting is not considered\n", __func__, inv.n_layer_nextn);
    } else {
        params.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
    }

    fit_advisor_probe probe(params);

    // the user's own arguments come first; their probe also reveals the model's devices for the allocations
    std::vector<fit_advisor_candidate> cands = { user_candidate(params) };
    const std::vector<std::string> device_bufts = fit_advisor_device_bufts(probe.run(cands[0]));
    const std::vector<std::string> device_names = fit_advisor_device_names(probe.run(cands[0]));
    std::vector<named_allocation> allocs = build_allocations(params, inv, device_bufts, cands[0]);
    for (const auto & na : allocs) {
        cands.push_back(na.alloc.to_candidate(inv, device_bufts, na.name));
    }

    // the built-in fitter's answer for the same arguments, for comparison
    std::vector<uint32_t> fitter_partition;
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
        if (status == COMMON_PARAMS_FIT_STATUS_SUCCESS && fit.n_gpu_layers >= 0) {
            // the fitter's placement priced beside the references, so its prediction can be read against a real run,
            // and its layer split seeds the search: it fills the fast card first, which the proportional seeds do not
            allocs.push_back({ "fit", allocation_from_candidate(inv, device_bufts, fit, (uint32_t) std::max(1, params.n_parallel)) });
            fitter_partition = allocs.back().alloc.layers_per_device;
        }
    }

    LOG_INF("%s: probing %zu candidates ...\n", __func__, cands.size());
    common_log_flush(common_log_main());

    print_table(cands, probe, params.n_batch, params);
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
        emit_if_requested(params, cands[1], "built-in fitter's");
        return 0;
    }

    // measure the devices for the types this model actually uses
    fit_advisor_measure_options mopts;
    mopts.weight_types = inv.matmul_types(0); // every type present, so no op is ever priced from nothing
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
    if (inv.is_moe() && inv.n_expert_used > 0 && inv.n_ff_exp > 0) {
        // expert matmuls are measured through mul_mat_id with the model's own routing and expert shape (gate/up)
        mopts.n_expert      = (int) inv.n_expert;
        mopts.n_expert_used = (int) inv.n_expert_used;
        mopts.moe_k = inv.n_embd;
        mopts.moe_m = inv.n_ff_exp;
        for (const auto & t : inv.tensors) {
            if (t.kind == FIT_ADVISOR_TENSOR_FFN_EXPS && std::find(mopts.moe_types.begin(), mopts.moe_types.end(), t.type) == mopts.moe_types.end()) {
                mopts.moe_types.push_back(t.type);
            }
        }
    }
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

    // the margin now only has to cover what the measurement cannot see, unless the user fixed it with --fit-target
    for (size_t d = 0; d < device_names.size(); d++) {
        for (size_t i = 0; i < devs.size(); i++) {
            if (device_names[d] == ggml_backend_dev_name(devs[i]) && meas[i]->runtime_overhead_bytes >= 0 && !params.fit_params_target_set) {
                const int64_t margin = meas[i]->runtime_overhead_bytes + FIT_ADVISOR_MARGIN_PAD;
                if (proj0.devices.size() > d && !proj0.devices[d].scratch_unknown) {
                    probe.set_margin(d, margin);
                    LOG_INF("%s: %s margin set to %.0f MiB: measured runtime overhead %.0f + %.0f MiB pad\n", __func__,
                        device_names[d].c_str(), margin / (1024.0 * 1024), meas[i]->runtime_overhead_bytes / (1024.0 * 1024),
                        FIT_ADVISOR_MARGIN_PAD / (1024.0 * 1024));
                }
            }
        }
    }
    common_log_flush(common_log_main());

    print_layer_costs(inv, devs, meas);

    // cost of every allocation under the workload
    fit_advisor_workload wl = [&]() {
        fit_advisor_workload w = fit_advisor_workload::preset(params.fit_advisor_workload);
        w.n_ubatch = (uint32_t) params.n_ubatch;
        w.use_mtp  = std::find(params.speculative.types.begin(), params.speculative.types.end(),
                               COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();
        w.expert_coverage    = params.fit_advisor_expert_coverage;
        w.pinned_cpu_weights = params.fit_advisor_pin_cpu_weights;
        w.op_offload_min_batch     = params.op_offload_min_batch;
        w.op_offload_min_batch_dev = params.op_offload_min_batch_dev;
        w.mtp_trained_depth = std::max<uint32_t>(1, inv.n_layer_nextn);
        w.mtp_accept  = params.fit_advisor_mtp_accept;
        w.mtp_decay   = params.fit_advisor_mtp_decay;
        return w;
    }();

    // transfers between every pair of devices, and the model's op counts per layer
    cache.ensure_pairs(devs, mopts.n_threads, params.fit_advisor_remeasure);

    std::vector<std::string> tensor_names;
    for (const auto & t : inv.tensors) {
        tensor_names.push_back(t.name);
    }
    const fit_advisor_graph_profile gp = probe.graph_profile(inv.n_layer + inv.n_layer_nextn, tensor_names);
    if (gp.ok) {
        uint32_t sum_tg = 0, sum_pp = 0;
        for (uint32_t x : gp.ops_per_layer_tg) sum_tg += x;
        for (uint32_t x : gp.ops_per_layer_pp) sum_pp += x;
        size_t n_used = 0, n_matmul = 0;
        for (const auto & u : gp.use_tg) { n_used += u.op != 0; n_matmul += u.is_matmul; }
        LOG_INF("%s: graph reads %zu of %zu weights, %zu through matmuls\n", __func__, n_used, gp.use_tg.size(), n_matmul);
        LOG_INF("%s: graph ops: %u nodes at batch 1 (%.1f per layer, %u global), %u nodes at batch %u (%.1f per layer, %u global)\n", __func__,
            gp.n_nodes_tg, gp.ops_per_layer_tg.empty() ? 0.0 : (double) sum_tg / gp.ops_per_layer_tg.size(), gp.ops_global_tg,
            gp.n_nodes_pp, gp.n_batch_pp, gp.ops_per_layer_pp.empty() ? 0.0 : (double) sum_pp / gp.ops_per_layer_pp.size(), gp.ops_global_pp);
    } else {
        LOG_WRN("%s: graph profile unavailable: %s\n", __func__, gp.error.c_str());
    }

    std::vector<fit_advisor_cost_device> cost_devs;
    std::vector<ggml_backend_dev_t>      cost_dev_handles;
    for (size_t d = 0; d < device_bufts.size(); d++) {
        fit_advisor_cost_device cd;
        cd.name   = device_bufts[d];
        cd.n_embd = inv.n_embd;
        ggml_backend_dev_t handle = nullptr;
        for (size_t i = 0; i < devs.size(); i++) {
            if (device_names[d] == ggml_backend_dev_name(devs[i])) {
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
    if (wl.use_mtp) {
        printf("  MTP drafting allowed: %.0f%% acceptance per draft position up to the trained depth %u (--mtp-accept), decaying by %.2f "
               "per position beyond it (--mtp-decay); the search chooses drafting and its depth per allocation (--no-mtp disables)\n",
            100.0 * wl.mtp_accept, wl.mtp_trained_depth, wl.mtp_decay);
    }
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
    fflush(stdout);

    // ---- the search, then an optional real load to measure what the projection missed and search again with that margin
    fit_advisor_search_options sopts;
    sopts.n_ctx        = params.n_ctx;
    if (!fitter_partition.empty()) {
        sopts.extra_partitions.push_back(fitter_partition);
    }
    if (params.n_ctx == 0) {
        // no context given: the model's default is its training context, which for large models fits nowhere. like
        // the fitter, halve from there until the experts-on-CPU (or all-on-device) reference fits, then say so
        const fit_advisor_projection & p_default = probe.run(cands[0]);
        uint32_t ctx = p_default.n_ctx_train;
        const uint32_t ctx_min = params.fit_params_min_ctx == UINT32_MAX ? 4096 : std::max<uint32_t>(256, (uint32_t) params.fit_params_min_ctx);
        const named_allocation & ref = allocs.back(); // the last reference allocation keeps the most on the CPU
        while (ctx > ctx_min) {
            fit_advisor_allocation a = ref.alloc;
            a.n_ctx = ctx;
            if (probe.run(a.to_candidate(inv, device_bufts, "ctx-" + std::to_string(ctx))).fits_all()) {
                break;
            }
            ctx = std::max(ctx_min, ctx / 2);
        }
        if (ctx != p_default.n_ctx_train) {
            LOG_WRN("%s: no -c given and the model's default context of %u does not fit; searching at %u (pass -c to choose)\n",
                __func__, p_default.n_ctx_train, ctx);
        } else {
            LOG_INF("%s: no -c given, searching at the model's default context of %u\n", __func__, ctx);
        }
        sopts.n_ctx = ctx;
    }
    sopts.max_slots    = wl.concurrency;
    sopts.anneal_iters = params.fit_advisor_search_iters;
    sopts.base_flash_attn    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO ? -1 : params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_ENABLED ? 1 : 0;
    sopts.base_no_kv_offload = params.no_kv_offload;
    sopts.search_flash_attn  = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO; // an explicit -fa is obeyed
    sopts.search_kv_offload  = !params.no_kv_offload;                                 // an explicit -nkvo is obeyed
    sopts.ubatch_options.clear();
    for (const auto & v : string_split<std::string>(params.fit_advisor_search_ubatch, ',')) {
        const int ub = std::atoi(v.c_str());
        if (ub <= 0) {
            LOG_ERR("%s: --search-ubatch: '%s' is not a positive integer\n", __func__, v.c_str());
            return 1;
        }
        sopts.ubatch_options.push_back((uint32_t) ub);
    }
    std::sort(sopts.ubatch_options.begin(), sopts.ubatch_options.end());
    sopts.ubatch_options.erase(std::unique(sopts.ubatch_options.begin(), sopts.ubatch_options.end()), sopts.ubatch_options.end());

    fit_advisor_search_result sr = search_and_report(params, inv, probe, device_bufts, gp, cost_devs, pair_table, wl, sopts);
    if (!sr.ok) {
        emit_if_requested(params, cands[1], "built-in fitter's"); // cands[1] is the fitter's own choice
        return 0;
    }

    if (params.fit_advisor_validate) {
        const uint32_t n_tokens = (uint32_t) std::max(0, params.fit_advisor_validate_tokens);
        fit_advisor_validate_result vr = fit_advisor_validate(params, sr.cand, sr.proj, n_tokens);
        fit_advisor_validate_print(vr);
        bool changed = false;
        if (vr.ok && vr.coverage_samples > 0 && params.fit_advisor_expert_coverage <= 0) {
            // the copies were priced on an assumed share; the validation measured it on real text
            const double assumed = wl.expert_coverage > 0 ? wl.expert_coverage : 0.6;
            LOG_INF("%s: expert coverage measured at %.2f (assumed %.2f)\n", __func__, vr.expert_coverage, assumed);
            if (std::fabs(vr.expert_coverage - assumed) > 0.1) {
                wl.expert_coverage = vr.expert_coverage;
                changed = true;
            }
        }
        if (vr.ok && vr.t_step_plain_us > 0) {
            // the plain step was timed; what it costs beyond the model is attributed to the ops that run away from
            // their layer's device, each of which is a scheduler split the kernel curves and link rates do not see
            const double plain_model = sr.cost.mtp_depth_step_us.empty() ? sr.cost.t_gen_step_us : sr.cost.mtp_depth_step_us[0];
            const double residual    = vr.t_step_plain_us - plain_model;
            if (sr.cost.n_excursions > 0) {
                const double extra = std::max(0.0, wl.excursion_extra_us + residual / sr.cost.n_excursions);
                LOG_INF("%s: plain step %.1f ms measured vs %.1f modelled with %u excursions: extra per excursion %.0f us (was %.0f)\n",
                    __func__, vr.t_step_plain_us * 1e-3, plain_model * 1e-3, sr.cost.n_excursions, extra, wl.excursion_extra_us);
                if (std::fabs(extra - wl.excursion_extra_us) * sr.cost.n_excursions > 0.02 * vr.t_step_plain_us) {
                    wl.excursion_extra_us = extra;
                    changed = true;
                }
            } else {
                LOG_INF("%s: plain step %.1f ms measured vs %.1f modelled, no excursions to attribute the difference to\n",
                    __func__, vr.t_step_plain_us * 1e-3, plain_model * 1e-3);
            }
        }
        if (vr.ok && vr.t_draft_us > 0 && sr.cost.mtp_depth_step_us.size() > 1) {
            // drafting was priced from kernels; the validation timed real steps. the extra per draft position is the
            // measured growth of the step over the plain step, less what the model already prices for that depth
            const double plain_model = sr.cost.mtp_depth_step_us[0];
            double extra_sum = 0;
            int    extra_n   = 0;
            std::string table;
            std::vector<double> by_depth(vr.t_verify_us.size(), 0.0);
            // the model's growth with depth at every timed depth: the scan only went as deep as it paid, so the deeper
            // depths are priced here with the same workload at that depth
            for (size_t d = 1; d < vr.t_verify_us.size(); d++) {
                if (vr.t_verify_us[d] <= 0) {
                    continue;
                }
                double p_all = 1.0;
                for (size_t k = 1; k <= d; k++) {
                    p_all *= wl.mtp_accept_at((uint32_t) k);
                }
                double prev_extra = d < wl.mtp_extra_by_depth_us.size() && wl.mtp_extra_by_depth_us[d] != 0
                    ? wl.mtp_extra_by_depth_us[d] : d * wl.mtp_extra_per_depth_us;
                double model_step;
                if (d < sr.cost.mtp_depth_step_us.size()) {
                    model_step = sr.cost.mtp_depth_step_us[d];
                } else {
                    // beyond the scan: the kernels' growth is near linear in the rows, extrapolate the last two depths
                    const size_t n = sr.cost.mtp_depth_step_us.size();
                    const double last = sr.cost.mtp_depth_step_us[n - 1], prev = sr.cost.mtp_depth_step_us[n - 2];
                    const double prev_extra_last = (n - 1) < wl.mtp_extra_by_depth_us.size() && wl.mtp_extra_by_depth_us[n - 1] != 0
                        ? wl.mtp_extra_by_depth_us[n - 1] : (n - 1) * wl.mtp_extra_per_depth_us;
                    const double prev_extra_prev = (n - 2) < wl.mtp_extra_by_depth_us.size() && wl.mtp_extra_by_depth_us[n - 2] != 0
                        ? wl.mtp_extra_by_depth_us[n - 2] : (n - 2) * wl.mtp_extra_per_depth_us;
                    const double slope = (last - prev_extra_last) - (prev - prev_extra_prev);
                    model_step = last - prev_extra_last + (d - (n - 1)) * slope;
                    prev_extra = 0;
                }
                // growth of the step with depth, without the terms a previous validation already added
                const double measured = vr.t_verify_us[d] + d * vr.t_draft_us - vr.t_step_plain_us;
                const double modelled = model_step - plain_model - prev_extra - (1.0 - p_all) * wl.mtp_rollback_us;
                by_depth[d] = measured - modelled; // may be negative where the kernel curves overprice this allocation
                extra_sum += (measured - modelled) / d;
                extra_n++;
                table += string_format(" d=%zu %.1f/%.1f", d, measured * 1e-3, modelled * 1e-3);
            }
            if (extra_n > 0) {
                const double extra = std::max(0.0, extra_sum / extra_n);
                wl.mtp_extra_by_depth_us = by_depth;
                LOG_INF("%s: drafting timed: plain step %.1f ms measured vs %.1f modelled; growth per depth measured/modelled ms:%s; "
                        "extra per draft position %.2f ms (was %.2f), rollback %.2f ms (was %.2f)\n", __func__, vr.t_step_plain_us * 1e-3, plain_model * 1e-3,
                        table.c_str(), extra * 1e-3, wl.mtp_extra_per_depth_us * 1e-3, vr.t_rollback_us * 1e-3, wl.mtp_rollback_us * 1e-3);
                bool differs = std::fabs(extra - wl.mtp_extra_per_depth_us) > 0.02 * vr.t_step_plain_us ||
                               std::fabs(vr.t_rollback_us - wl.mtp_rollback_us) > 0.02 * vr.t_step_plain_us;
                for (size_t d = 1; d < by_depth.size(); d++) {
                    differs = differs || std::fabs(by_depth[d]) > 0.02 * vr.t_step_plain_us;
                }
                if (differs) {
                    wl.mtp_extra_per_depth_us = extra;
                    wl.mtp_rollback_us        = vr.t_rollback_us;
                    changed = true;
                }
            }
        }
        if (vr.ok && !params.fit_params_target_set) {
            for (size_t d = 0; d < vr.devices.size() && d < sr.proj.devices.size(); d++) {
                const int64_t suggested = vr.suggested_margin(d);
                const int64_t current   = sr.proj.devices[d].margin;
                if (!sr.proj.devices[d].scratch_unknown && std::llabs(suggested - current) > 64ll * 1024 * 1024) {
                    LOG_INF("%s: %s margin %.0f -> %.0f MiB from the validation run\n", __func__, sr.proj.devices[d].name.c_str(),
                        current / (1024.0 * 1024), suggested / (1024.0 * 1024));
                    probe.set_margin(d, suggested);
                    changed = true;
                }
            }
        }
        {
            if (changed) {
                LOG_INF("%s: searching again with the validated margins, coverage, excursion and drafting costs ...\n", __func__);
                fit_advisor_search_result sr2 = search_and_report(params, inv, probe, device_bufts, gp, cost_devs, pair_table, wl, sopts);
                if (sr2.ok) {
                    fit_advisor_validate_result vr2 = fit_advisor_validate(params, sr2.cand, sr2.proj, n_tokens);
                    fit_advisor_validate_print(vr2);
                    bool within = vr2.ok;
                    for (size_t d = 0; d < vr2.devices.size() && within; d++) {
                        within = vr2.devices[d].unmodelled() <= vr2.devices[d].margin;
                    }
                    if (within) {
                        sr = sr2;
                    } else {
                        LOG_WRN("%s: the allocation found with the validated margins did not validate, keeping the first one\n", __func__);
                    }
                } else {
                    LOG_WRN("%s: no feasible allocation with the validated margins, keeping the first one\n", __func__);
                }
            } else {
                LOG_INF("%s: validated margins match the ones used, the allocation stands\n", __func__);
            }
        }
    }

    if (params.fit_advisor_verify) {
        fit_advisor_verify_allocation(params, inv, sr.alloc, device_bufts, sr.cand);
    }

    emit_if_requested(params, sr.cand, "searched");
    return 0;
}
