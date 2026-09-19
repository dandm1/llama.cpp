#include "cost.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <map>

fit_advisor_workload fit_advisor_workload::preset(const std::string & name) {
    fit_advisor_workload w;
    if (name == "chat" || name.empty()) {
        return w;
    }
    if (name == "rag") {
        w.prompt_tokens = 16384;
        w.gen_tokens    = 256;
        return w;
    }
    if (name == "batch") {
        w.prompt_tokens = 1024;
        w.gen_tokens    = 256;
        w.concurrency   = 8;
        w.throughput    = true;
        return w;
    }
    if (name == "agent") {
        w.prompt_tokens = 8192;
        w.gen_tokens    = 2048;
        return w;
    }
    return w;
}

// the matmul curve has three points: batch 1 and 4 on the large weight, batch n_pp on the small one
// interpolate per-token seconds per byte log-linearly in the batch size, extrapolating the last segment
double fit_advisor_s_per_byte(const fit_advisor_matmul_rate & r, uint32_t batch) {
    if (!r.supported || r.s_per_byte_b1 <= 0) {
        return 0;
    }
    // per-token cost at each point
    const double c1 = r.s_per_byte_b1;
    const double c4 = r.s_per_byte_b4 > 0 ? r.s_per_byte_b4 / 4 : c1;
    const double np = r.n_batch_pp > 0 ? r.n_batch_pp : 512;
    const double cp = r.s_per_byte_bpp > 0 ? r.s_per_byte_bpp / np : c4;

    if (batch <= 1) {
        return c1;
    }
    const double lb = std::log((double) batch);
    if (batch <= 4) {
        const double f = lb / std::log(4.0);
        return std::exp(std::log(c1) + f * (std::log(c4) - std::log(c1)));
    }
    if ((double) batch <= np) {
        const double f = (lb - std::log(4.0)) / (std::log(np) - std::log(4.0));
        return std::exp(std::log(c4) + f * (std::log(cp) - std::log(c4)));
    }
    // beyond the prompt batch the op is compute bound: per-token cost stays flat
    return cp;
}

namespace {

// expert tensors: at batch b each token routes to n_used of n_expert experts, so an expert sees on average
// b * n_used / n_expert tokens. below one token per expert only that fraction of the experts is touched at all;
// above it every expert is touched and runs at that smaller effective batch
struct expert_view {
    double frac_touched = 1; // share of the tensor's bytes read
    double batch        = 1; // tokens each touched expert processes
};

expert_view expert_batch(const fit_advisor_inventory & inv, const fit_advisor_tensor & t, uint32_t batch) {
    expert_view v;
    v.batch = batch;
    if (t.kind != FIT_ADVISOR_TENSOR_FFN_EXPS || inv.n_expert == 0 || inv.n_expert_used == 0) {
        return v;
    }
    const double per_expert = (double) batch * inv.n_expert_used / inv.n_expert;
    v.frac_touched = std::min(1.0, per_expert);
    v.batch        = std::max(1.0, per_expert);
    return v;
}

// time for one op over a tensor: per-token cost times tokens plus the fixed overhead
// weights on the CPU above the device's offload threshold are copied to the device and computed there
double tensor_us(const fit_advisor_inventory & inv, const fit_advisor_tensor & t, int dev_idx, int layer_dev,
                 const std::vector<fit_advisor_cost_device> & devices, uint32_t batch, std::string & error) {
    const fit_advisor_cost_device * dev = &devices[dev_idx < 0 ? devices.size() - 1 : (size_t) dev_idx];
    GGML_UNUSED(layer_dev);

    double copy_us = 0;
    if (dev->is_cpu) {
        // op offload: the scheduler hands an op with a CPU-resident weight to the first device that wants it at this
        // batch size, whatever layer it belongs to; the weight is copied there for every ubatch
        for (const auto & d : devices) {
            if (!d.is_cpu && d.meas && d.offload_min_batch > 0 && batch >= (uint32_t) d.offload_min_batch) {
                if (d.meas->copy.h2d_gb_s > 0) {
                    copy_us = t.nbytes / (d.meas->copy.h2d_gb_s * 1e9) * 1e6;
                }
                dev = &d;
                break;
            }
        }
    }

    const auto it = dev->meas ? dev->meas->matmul.find(ggml_type_name(t.type)) : dev->meas->matmul.end();
    if (!dev->meas || it == dev->meas->matmul.end() || !it->second.supported) {
        if (t.nbytes >= 1024 * 1024) {
            error = std::string("no measured rate for ") + ggml_type_name(t.type) + " on " + dev->name;
        }
        return copy_us; // norms and biases: negligible
    }
    const expert_view ev = expert_batch(inv, t, batch);
    const uint32_t b_eff = (uint32_t) std::lround(ev.batch);
    return copy_us + t.nbytes * ev.frac_touched * fit_advisor_s_per_byte(it->second, b_eff) * 1e6 * ev.batch;
}

} // namespace

fit_advisor_cost fit_advisor_cost_estimate(const fit_advisor_inventory & inv, const fit_advisor_allocation & alloc,
                                           const fit_advisor_projection & proj, const fit_advisor_graph_profile & gp,
                                           const std::vector<fit_advisor_cost_device> & devices, const fit_advisor_pair_table & pairs,
                                           const fit_advisor_workload & wl) {
    fit_advisor_cost c;
    if (!proj.ok || devices.empty() || alloc.tensor_device.size() != inv.tensors.size()) {
        c.error = "missing projection, devices or allocation";
        return c;
    }
    const uint32_t n_layer_all = inv.n_layer + inv.n_layer_nextn;
    const uint32_t batch_gen   = std::max<uint32_t>(1, std::min(wl.concurrency, alloc.n_slots));

    // effective context: total across slots, each slot filled to prompt plus half the generation on average
    const uint32_t n_ctx_total = alloc.n_ctx > 0 ? alloc.n_ctx : proj.n_ctx_train;
    const uint32_t n_ctx_slot  = std::max<uint32_t>(1, n_ctx_total / std::max<uint32_t>(1, alloc.n_slots));
    const double   fill_frac   = std::min(1.0, (wl.prompt_tokens + wl.gen_tokens / 2.0) / n_ctx_slot);

    // one decode step: every weight once for the batch, attention over each active slot's KV, boundaries
    auto step_us = [&](uint32_t batch, double & weights, double & attn, double & overhead, double & boundary) {
        weights = attn = overhead = boundary = 0;
        for (size_t i = 0; i < inv.tensors.size(); i++) {
            const auto & t = inv.tensors[i];
            if (t.layer >= (int32_t) inv.n_layer && !wl.use_mtp) {
                continue; // MTP layer, not loaded and not executed
            }
            const int layer_dev = t.layer >= 0 ? alloc.layer_device((uint32_t) t.layer, n_layer_all)
                                               : (t.kind == FIT_ADVISOR_TENSOR_TOKEN_EMBD ? fit_advisor_allocation::DEV_CPU
                                                                                            : alloc.layer_device(n_layer_all, n_layer_all));
            if (t.kind == FIT_ADVISOR_TENSOR_TOKEN_EMBD) {
                continue; // a row lookup, not a matmul
            }
            weights += tensor_us(inv, t, alloc.tensor_device[i], layer_dev, devices, batch, c.error);
        }

        // attention: the KV bytes each device holds for this candidate, scaled by fill and active slots
        for (size_t d = 0; d < proj.devices.size() && d < devices.size(); d++) {
            const auto & pd = proj.devices[d];
            const auto * m  = devices[d].meas;
            if (!m || pd.context == 0) {
                continue;
            }
            double best = 0;
            for (const auto & [key, r] : m->attn) {
                best = std::max({ best, r.kv_bytes_per_s_fa, r.kv_bytes_per_s_nofa });
            }
            if (best > 0) {
                attn += pd.context * fill_frac * ((double) batch_gen / alloc.n_slots) / best * 1e6;
            }
        }
        {
            const auto * m = devices.back().meas;
            if (m && proj.host.context > 0) {
                double best = 0;
                for (const auto & [key, r] : m->attn) {
                    best = std::max({ best, r.kv_bytes_per_s_fa, r.kv_bytes_per_s_nofa });
                }
                if (best > 0) {
                    attn += proj.host.context * fill_frac * ((double) batch_gen / alloc.n_slots) / best * 1e6;
                }
            }
        }

        // per-op overhead: every graph node of a layer costs the fixed per-op time of the layer's device
        const bool pp = batch > 4;
        const std::vector<uint32_t> & ops = pp ? gp.ops_per_layer_pp : gp.ops_per_layer_tg;
        for (uint32_t il = 0; il < n_layer_all && il < ops.size(); il++) {
            if (il >= inv.n_layer && !wl.use_mtp) {
                continue;
            }
            const int d = alloc.layer_device(il, n_layer_all);
            const auto & cd = devices[d < 0 ? devices.size() - 1 : (size_t) d];
            if (cd.meas) {
                overhead += ops[il] * cd.meas->op_overhead_us;
            }
        }
        {
            const int d = alloc.layer_device(n_layer_all, n_layer_all);
            const auto & cd = devices[d < 0 ? devices.size() - 1 : (size_t) d];
            if (cd.meas) {
                overhead += (pp ? gp.ops_global_pp : gp.ops_global_tg) * cd.meas->op_overhead_us;
            }
        }

        // boundaries: each device change along the layer sequence moves the activation (batch x n_embd f32),
        // and each group of tensors away from their layer moves it there and back
        auto hop_us = [&](int from, int to) -> double {
            const size_t a = from < 0 ? devices.size() - 1 : (size_t) from;
            const size_t b = to   < 0 ? devices.size() - 1 : (size_t) to;
            if (a == b || a >= pairs.size() || b >= pairs[a].size()) {
                return 0;
            }
            const fit_advisor_pair_rate & r = pairs[a][b];
            const double bytes = (double) batch * inv.n_embd * sizeof(float);
            return r.latency_us + (r.gb_s > 0 ? bytes / (r.gb_s * 1e9) * 1e6 : 0);
        };
        int prev = alloc.layer_device(0, n_layer_all);
        for (uint32_t il = 1; il <= n_layer_all; il++) {
            const int d = alloc.layer_device(il, n_layer_all);
            if (d != prev) {
                boundary += hop_us(prev, d);
            }
            prev = d;
        }
        std::map<std::pair<int, int>, int> away; // (layer, device) pairs with tensors away from home
        for (size_t i = 0; i < inv.tensors.size(); i++) {
            const auto & t = inv.tensors[i];
            if (t.layer < 0) {
                continue;
            }
            const int home = alloc.layer_device((uint32_t) t.layer, n_layer_all);
            if (alloc.tensor_device[i] != home) {
                away[{ t.layer, alloc.tensor_device[i] }]++;
            }
        }
        for (const auto & [key, n] : away) {
            const int home = alloc.layer_device((uint32_t) key.first, n_layer_all);
            boundary += hop_us(home, key.second) + hop_us(key.second, home);
        }
    };

    step_us(batch_gen, c.step_weights_us, c.step_attn_us, c.step_overhead_us, c.step_boundary_us);
    c.t_gen_step_us = c.step_weights_us + c.step_attn_us + c.step_overhead_us + c.step_boundary_us;

    // prompt: ubatches of wl.n_ubatch tokens, attention grows with the prefix, approximated at half fill
    {
        double w, a, o, b;
        const uint32_t n_ub = std::max<uint32_t>(1, wl.n_ubatch);
        step_us(n_ub, w, a, o, b);
        const double n_steps = std::ceil((double) wl.prompt_tokens / n_ub);
        // attention during the prompt sees on average half the prompt, per ubatch of n_ub query rows:
        // scale the batch-1 attention figure by rows and by the prompt's share of the fill
        const double a_prompt = a * n_ub * (wl.prompt_tokens / 2.0) / std::max(1.0, fill_frac * n_ctx_slot);
        c.t_prompt_us = n_steps * (w + a_prompt + o + b);
    }

    c.t_request_us = c.t_prompt_us + wl.gen_tokens * c.t_gen_step_us;
    c.gen_tokens_per_s    = c.t_gen_step_us > 0 ? batch_gen * 1e6 / c.t_gen_step_us : 0;
    c.prompt_tokens_per_s = c.t_prompt_us > 0 ? wl.prompt_tokens * 1e6 / c.t_prompt_us : 0;
    c.ok = true;
    return c;
}
