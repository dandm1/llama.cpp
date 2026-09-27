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
    if (name == "gen") {
        // generation speed above all: a short prompt and a long answer
        w.prompt_tokens = 256;
        w.gen_tokens    = 2048;
        return w;
    }
    return w;
}

// the matmul curve has three points: batch 1 and 4 on the large weight, batch n_pp on the small one
// interpolate per-token seconds per byte log-linearly in the batch size, extrapolating the last segment
// per-token seconds at a batch from measured (batch -> seconds for the whole batch) points: log-linear between
// bracketing points, the first point below its batch, flat per token beyond the last (compute bound)
static double interp_per_token(const std::map<int, double> & points, uint32_t batch) {
    if (points.empty()) {
        return 0;
    }
    auto per_token = [](const std::pair<const int, double> & p) { return p.second / p.first; };
    auto hi = points.lower_bound((int) batch);
    if (hi == points.end()) {
        return per_token(*std::prev(points.end()));
    }
    if (hi == points.begin() || hi->first == (int) batch) {
        return per_token(*hi);
    }
    auto lo = std::prev(hi);
    const double f = (std::log((double) batch) - std::log((double) lo->first)) / (std::log((double) hi->first) - std::log((double) lo->first));
    return std::exp(std::log(per_token(*lo)) + f * (std::log(per_token(*hi)) - std::log(per_token(*lo))));
}

double fit_advisor_s_per_byte(const fit_advisor_matmul_rate & r, uint32_t batch) {
    if (!r.supported || r.s_per_byte_b1 <= 0) {
        return 0;
    }
    if (r.points.size() >= 2) {
        return interp_per_token(r.points, std::max<uint32_t>(1, batch));
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

double fit_advisor_workload::tokens_per_step() const {
    if (!use_mtp || mtp_draft_n == 0) {
        return 1.0;
    }
    // acceptances treated as independent: the k-th draft token counts only if every earlier one was accepted
    double ret = 1.0, pk = 1.0;
    for (uint32_t k = 0; k < mtp_draft_n; k++) {
        pk  *= std::min(1.0, std::max(0.0, mtp_accept));
        ret += pk;
    }
    return ret;
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

// memory rate of a device: the best measured weight-streaming rate stands in for its bandwidth
double device_bandwidth(const fit_advisor_cost_device & d) {
    double best = 0;
    if (d.meas) {
        for (const auto & [type, r] : d.meas->matmul) {
            best = std::max(best, r.bytes_per_s);
        }
    }
    return best;
}

// time for one op over a tensor: per-token cost times tokens; matmuls on the measured curve, other ops by the
// activation bytes they touch. weights on the CPU above a device's offload threshold are copied there and run there.
} // namespace

int fit_advisor_offload_taker(const std::vector<fit_advisor_cost_device> & devices, uint32_t batch, int home_idx, const fit_advisor_workload * wl) {
    auto takes = [&](size_t d) {
        const auto & dev = devices[d];
        const int32_t min_batch = wl ? wl->offload_min_for(d, dev.offload_min_batch) : dev.offload_min_batch;
        return !dev.is_cpu && dev.meas && min_batch > 0 && batch >= (uint32_t) min_batch;
    };
    if (home_idx >= 0 && (size_t) home_idx < devices.size() && takes((size_t) home_idx)) {
        return home_idx;
    }
    for (size_t d = 0; d < devices.size(); d++) {
        if (takes(d)) {
            return (int) d;
        }
    }
    return -1;
}

namespace {

double tensor_us(const fit_advisor_inventory & inv, const fit_advisor_tensor & t, const fit_advisor_tensor_use & use,
                 int dev_idx, const std::vector<fit_advisor_cost_device> & devices, uint32_t batch, std::string & error, int home_idx,
                 const fit_advisor_workload * wl) {
    if (use.op == 0) {
        return 0; // not read by the graph (e.g. an MTP tensor with MTP off)
    }
    const fit_advisor_cost_device * dev = &devices[dev_idx < 0 ? devices.size() - 1 : (size_t) dev_idx];

    double copy_us = 0;
    if (dev->is_cpu) {
        // op offload: the scheduler hands an op with a CPU-resident weight to a device that wants it at this batch
        // size, preferring the device that holds the op's activations, i.e. the layer's home, and otherwise the
        // first willing one; the weight is copied there for every ubatch
        const int taker = fit_advisor_offload_taker(devices, batch, home_idx, wl);
        if (taker >= 0) {
            const fit_advisor_cost_device * target = &devices[taker];
            const double rate = (wl && wl->pinned_cpu_weights && target->meas->copy.h2d_pinned_gb_s > 0)
                ? target->meas->copy.h2d_pinned_gb_s : target->meas->copy.h2d_gb_s;
            if (rate > 0) {
                // expert stacks: only the experts the batch routes to are copied
                double share = 1.0;
                if (t.kind == FIT_ADVISOR_TENSOR_FFN_EXPS && inv.n_expert > 0 && inv.n_expert_used > 0) {
                    const double uniform = 1.0 - std::pow(1.0 - (double) inv.n_expert_used / inv.n_expert, (double) batch);
                    share = wl && wl->expert_coverage > 0 ? wl->expert_coverage : std::min(uniform, batch >= 256 ? 0.6 : 1.0);
                }
                copy_us = t.nbytes * share / (rate * 1e9) * 1e6;
            }
            dev = target;
        }
    }
    if (!dev->meas) {
        return copy_us;
    }

    if (use.is_matmul && t.kind == FIT_ADVISOR_TENSOR_FFN_EXPS) {
        // measured through mul_mat_id with this model's routing: seconds per stack byte for the whole ubatch,
        // interpolated per token in the batch, so the routed fraction and the gather/sort work are inside the number
        const auto it = dev->meas->moe.find(ggml_type_name(t.type));
        if (it != dev->meas->moe.end() && it->second.supported && it->second.n_expert == (int) inv.n_expert
            && it->second.n_expert_used == (int) inv.n_expert_used && !it->second.points.empty()) {
            const uint32_t b = std::max<uint32_t>(1, batch);
            return copy_us + t.nbytes * interp_per_token(it->second.points, b) * b * 1e6;
        }
    }
    if (use.is_matmul) {
        const auto it = dev->meas->matmul.find(ggml_type_name(t.type));
        if (it == dev->meas->matmul.end() || !it->second.supported || it->second.bytes_per_s <= 0) {
            error = std::string("no measured rate for ") + ggml_type_name(t.type) + " on " + dev->name;
            // fall back to the device's best rate so the cost is at least not zero
            const double bw = device_bandwidth(*dev);
            const expert_view ev = expert_batch(inv, t, batch);
            return copy_us + (bw > 0 ? t.nbytes * ev.frac_touched / bw * 1e6 : 0) + dev->meas->op_overhead_us;
        }
        const expert_view ev = expert_batch(inv, t, batch);
        const uint32_t b_eff = (uint32_t) std::lround(ev.batch);
        return copy_us + t.nbytes * ev.frac_touched * fit_advisor_s_per_byte(it->second, b_eff) * 1e6 * ev.batch;
    }

    // a norm, scale, conv or recurrent-state op: streams its activations once per ubatch, plus the per-op cost
    const double bw = device_bandwidth(*dev);
    return copy_us + dev->meas->op_overhead_us + (bw > 0 ? (double) use.act_bytes / bw * 1e6 : 0);
}

} // namespace

double fit_advisor_tensor_cost_us(const fit_advisor_inventory & inv, const fit_advisor_tensor & t, const fit_advisor_tensor_use & use,
                                  int dev_idx, const std::vector<fit_advisor_cost_device> & devices, uint32_t batch, int home_idx,
                                  const fit_advisor_workload * wl) {
    std::string err;
    return tensor_us(inv, t, use, dev_idx, devices, batch, err, home_idx, wl);
}

bool fit_advisor_tensor_cost_known(const fit_advisor_inventory & inv, const fit_advisor_tensor & t, const fit_advisor_tensor_use & use,
                                   int dev_idx, const std::vector<fit_advisor_cost_device> & devices) {
    GGML_UNUSED(inv);
    if (use.op == 0) {
        return true; // unused: a known zero
    }
    const fit_advisor_cost_device & dev = devices[dev_idx < 0 ? devices.size() - 1 : (size_t) dev_idx];
    if (!dev.meas) {
        return false;
    }
    if (use.is_matmul) {
        const auto it = dev.meas->matmul.find(ggml_type_name(t.type));
        return it != dev.meas->matmul.end() && it->second.supported && it->second.bytes_per_s > 0;
    }
    return dev.meas->op_overhead_us > 0 && device_bandwidth(dev) > 0;
}

double fit_advisor_tensor_request_us(const fit_advisor_inventory & inv, const fit_advisor_graph_profile & gp, size_t tensor_idx, int dev_idx,
                                     const std::vector<fit_advisor_cost_device> & devices, const fit_advisor_workload & wl, uint32_t n_slots, int home_idx) {
    const uint32_t batch_gen = std::max<uint32_t>(1, std::min(wl.concurrency, n_slots));
    const uint32_t n_ub      = std::max<uint32_t>(1, wl.n_ubatch);
    const double n_pp_steps  = std::ceil((double) wl.prompt_tokens / n_ub);
    const fit_advisor_tensor & t = inv.tensors[tensor_idx];
    const fit_advisor_tensor_use & u_pp = tensor_idx < gp.use_pp.size() ? gp.use_pp[tensor_idx] : fit_advisor_tensor_use{};
    const double pp_us = n_pp_steps * fit_advisor_tensor_cost_us(inv, t, u_pp, dev_idx, devices, n_ub, home_idx, &wl);

    const bool is_mtp = t.layer >= (int32_t) inv.n_layer;
    if (is_mtp && !wl.use_mtp) {
        return pp_us;
    }
    // generation: without drafting one step per token; with drafting a step verifies 1 + draft tokens in one batch
    // through the trunk and runs the MTP layer once per draft token, yielding tokens_per_step() tokens
    const double n_steps = wl.gen_tokens / wl.tokens_per_step();
    const uint32_t batch_step = is_mtp ? batch_gen : batch_gen * (wl.use_mtp ? 1 + wl.mtp_draft_n : 1);
    const fit_advisor_tensor_use & u_gen = batch_step > 4
        ? u_pp : (tensor_idx < gp.use_tg.size() ? gp.use_tg[tensor_idx] : fit_advisor_tensor_use{});
    const double runs_per_step = is_mtp ? (double) wl.mtp_draft_n : 1.0;
    return n_steps * runs_per_step * fit_advisor_tensor_cost_us(inv, t, u_gen, dev_idx, devices, batch_step, home_idx, &wl) + pp_us;
}

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
    // sel: 0 = trunk layers only, 1 = MTP layers only (one draft run), 2 = everything loaded in one graph
    enum { SEL_TRUNK = 0, SEL_MTP = 1, SEL_ALL = 2 };
    auto layer_selected = [&](int32_t layer, int sel) {
        const bool is_mtp = layer >= (int32_t) inv.n_layer;
        if (is_mtp && !wl.use_mtp) {
            return false; // MTP layer, not loaded and not executed
        }
        return sel == SEL_ALL || (sel == SEL_MTP) == is_mtp;
    };
    auto step_us = [&](uint32_t batch, int sel, double & weights, double & attn, double & overhead, double & boundary) {
        weights = attn = overhead = boundary = 0;
        const std::vector<fit_advisor_tensor_use> & uses = batch > 4 ? gp.use_pp : gp.use_tg;
        for (size_t i = 0; i < inv.tensors.size(); i++) {
            const auto & t = inv.tensors[i];
            if (t.layer >= 0 ? !layer_selected(t.layer, sel) : sel == SEL_MTP) {
                continue; // global tensors (embeddings, output) belong to the trunk graph
            }
            if (t.kind == FIT_ADVISOR_TENSOR_TOKEN_EMBD) {
                continue; // a row lookup, not a matmul
            }
            const fit_advisor_tensor_use use = i < uses.size() ? uses[i] : fit_advisor_tensor_use{};
            const int home = t.layer >= 0 ? alloc.layer_device((uint32_t) t.layer, n_layer_all) : alloc.layer_device(n_layer_all, n_layer_all);
            weights += tensor_us(inv, t, use, alloc.tensor_device[i], devices, batch, c.error, home, &wl);
        }

        // attention: the KV bytes each device holds for this candidate, scaled by fill and active slots; the MTP layers
        // hold their share of the cache, so a draft run attends over that share and the trunk over the rest
        const double attn_share = sel == SEL_ALL ? 1.0
            : sel == SEL_MTP ? (double) inv.n_layer_nextn / std::max<uint32_t>(1, n_layer_all)
            : (wl.use_mtp ? (double) inv.n_layer / std::max<uint32_t>(1, n_layer_all) : 1.0);
        // the attention path: flash attention when the allocation asks for it or leaves it to llama.cpp (which enables
        // it wherever the backend supports it), the explicit path when it is off or unsupported on the device
        auto attn_rate = [&](const fit_advisor_device_measurements * m) {
            double fa = 0, nofa = 0;
            for (const auto & [key, r] : m->attn) {
                fa   = std::max(fa,   r.supported_fa   ? r.kv_bytes_per_s_fa   : 0.0);
                nofa = std::max(nofa, r.supported_nofa ? r.kv_bytes_per_s_nofa : 0.0);
            }
            if (alloc.flash_attn == 0) {
                return nofa > 0 ? nofa : fa;
            }
            return fa > 0 ? fa : nofa;
        };
        for (size_t d = 0; d < proj.devices.size() && d < devices.size(); d++) {
            const auto & pd = proj.devices[d];
            const auto * m  = devices[d].meas;
            if (!m || pd.context == 0) {
                continue;
            }
            const double rate = attn_rate(m);
            if (rate > 0) {
                attn += attn_share * pd.context * fill_frac * ((double) batch_gen / alloc.n_slots) / rate * 1e6;
            }
        }
        {
            const auto * m = devices.back().meas;
            if (m && proj.host.context > 0) {
                const double rate = attn_rate(m);
                if (rate > 0) {
                    attn += attn_share * proj.host.context * fill_frac * ((double) batch_gen / alloc.n_slots) / rate * 1e6;
                }
            }
        }

        // per-op overhead: every graph node of a layer costs the fixed per-op time of the layer's device
        const bool pp = batch > 4;
        const std::vector<uint32_t> & ops = pp ? gp.ops_per_layer_pp : gp.ops_per_layer_tg;
        for (uint32_t il = 0; il < n_layer_all && il < ops.size(); il++) {
            if (!layer_selected((int32_t) il, sel)) {
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
            if (r.split_n_batch_pp > 0) {
                // measured through the scheduler: half a round trip per hop, plus bandwidth for the larger activation
                const bool pp = batch > 4;
                const double split = pp ? r.split_us_bpp : r.split_us_b1;
                const double extra = pp ? std::max(0.0, bytes - (double) r.split_bytes_bpp) : 0;
                return 0.5 * split + (r.gb_s > 0 ? extra / (r.gb_s * 1e9) * 1e6 : 0);
            }
            const double launch = devices[b].meas ? devices[b].meas->launch_us : 0;
            return r.latency_us + (r.gb_s > 0 ? bytes / (r.gb_s * 1e9) * 1e6 : 0) + launch;
        };
        int prev = alloc.layer_device(0, n_layer_all);
        for (uint32_t il = 1; il <= n_layer_all; il++) {
            const int d = alloc.layer_device(il, n_layer_all);
            if (d != prev) {
                boundary += hop_us(prev, d);
            }
            prev = d;
        }
        // excursions: tensors away from their layer's device, in graph order; consecutive away ops on the same device
        // share one excursion, anything else in between starts a new one. each excursion moves its first op's inputs
        // out and its last op's output back
        struct away_op { int node_idx; int dev; int home; size_t act_bytes; };
        std::vector<away_op> aways;
        for (size_t i = 0; i < inv.tensors.size() && i < uses.size(); i++) {
            const auto & t = inv.tensors[i];
            if (t.layer < 0 || uses[i].op == 0 || !layer_selected(t.layer, sel)) {
                continue;
            }
            const int home = alloc.layer_device((uint32_t) t.layer, n_layer_all);
            const int dev  = alloc.tensor_device[i];
            if (dev == home) {
                continue;
            }
            if (dev == fit_advisor_allocation::DEV_CPU) {
                const int taker = fit_advisor_offload_taker(devices, batch, home, &wl);
                if (taker == home) {
                    continue; // offloaded to the layer's own device, no split
                }
                if (taker >= 0) {
                    aways.push_back({ uses[i].node_idx, taker, home, uses[i].act_bytes }); // a trip to the taking device
                    continue;
                }
            }
            aways.push_back({ uses[i].node_idx, dev, home, uses[i].act_bytes });
        }
        std::sort(aways.begin(), aways.end(), [](const away_op & a, const away_op & b) { return a.node_idx < b.node_idx; });
        for (size_t k = 0; k < aways.size();) {
            size_t j = k;
            while (j + 1 < aways.size() && aways[j + 1].dev == aways[k].dev && aways[j + 1].node_idx <= aways[j].node_idx + 2) {
                j++;
            }
            // hop_us prices the round trip with the standard activation; add the op's own activation bytes over the link
            boundary += hop_us(aways[k].home, aways[k].dev) + hop_us(aways[k].dev, aways[k].home);
            const size_t a = aways[k].home < 0 ? devices.size() - 1 : (size_t) aways[k].home;
            const size_t b = aways[k].dev  < 0 ? devices.size() - 1 : (size_t) aways[k].dev;
            if (a < pairs.size() && b < pairs[a].size() && pairs[a][b].gb_s > 0) {
                boundary += (double) (aways[k].act_bytes + aways[j].act_bytes) / (pairs[a][b].gb_s * 1e9) * 1e6;
            }
            k = j + 1;
        }
    };

    c.tokens_per_step = wl.tokens_per_step();
    if (wl.use_mtp && inv.n_layer_nextn > 0 && wl.mtp_draft_n > 0) {
        // drafting: the trunk verifies 1 + draft tokens per slot in one batch, then the MTP layer runs once per
        // draft token at the generation batch; the step yields tokens_per_step tokens per slot
        step_us(batch_gen * (1 + wl.mtp_draft_n), SEL_TRUNK, c.step_weights_us, c.step_attn_us, c.step_overhead_us, c.step_boundary_us);
        double w, a, o, b;
        step_us(batch_gen, SEL_MTP, w, a, o, b);
        c.t_mtp_draft_us = w + a + o + b;
        c.t_gen_step_us  = c.step_weights_us + c.step_attn_us + c.step_overhead_us + c.step_boundary_us
                         + wl.mtp_draft_n * c.t_mtp_draft_us;
    } else {
        step_us(batch_gen, SEL_ALL, c.step_weights_us, c.step_attn_us, c.step_overhead_us, c.step_boundary_us);
        c.t_gen_step_us = c.step_weights_us + c.step_attn_us + c.step_overhead_us + c.step_boundary_us;
    }

    // prompt: ubatches of wl.n_ubatch tokens, attention grows with the prefix, approximated at half fill
    {
        double w, a, o, b;
        const uint32_t n_ub = std::max<uint32_t>(1, wl.n_ubatch);
        step_us(n_ub, SEL_ALL, w, a, o, b); // the draft context processes the prompt as well
        const double n_steps = std::ceil((double) wl.prompt_tokens / n_ub);
        // attention during the prompt sees on average half the prompt, per ubatch of n_ub query rows:
        // scale the batch-1 attention figure by rows and by the prompt's share of the fill
        const double a_prompt = a * n_ub * (wl.prompt_tokens / 2.0) / std::max(1.0, fill_frac * n_ctx_slot);
        c.t_prompt_us = n_steps * (w + a_prompt + o + b);
    }

    c.t_request_us = c.t_prompt_us + wl.gen_tokens / c.tokens_per_step * c.t_gen_step_us;
    c.gen_tokens_per_s    = c.t_gen_step_us > 0 ? batch_gen * c.tokens_per_step * 1e6 / c.t_gen_step_us : 0;
    c.prompt_tokens_per_s = c.t_prompt_us > 0 ? wl.prompt_tokens * 1e6 / c.t_prompt_us : 0;
    c.ok = true;
    return c;
}
