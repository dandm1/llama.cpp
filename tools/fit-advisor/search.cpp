#include "search.h"
#include "solve.h"

#include "ggml.h"

#include "log.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <map>
#include <numeric>

namespace {

constexpr int64_t UNIT = 1024 * 1024; // memory granularity of the capacities

std::string cell_name(const std::vector<uint32_t> & part, uint32_t ub, uint32_t slots) {
    std::string n = "search";
    for (size_t d = 0; d < part.size(); d++) {
        n += (d ? "/" : "-") + std::to_string(part[d]);
    }
    return n + "-ub" + std::to_string(ub) + "-np" + std::to_string(slots);
}

std::string alloc_name(const fit_advisor_allocation & a) {
    std::string ret = cell_name(a.layers_per_device, a.n_ubatch, a.n_slots);
    for (const auto & [il, dev] : a.layer_home) {
        ret += "-L" + std::to_string(il) + (dev < 0 ? "cpu" : std::to_string(dev));
    }
    if (a.draft_mtp) {
        ret += "-mtp";
    }
    if (a.flash_attn >= 0) {
        ret += a.flash_attn ? "-fa" : "-nofa";
    }
    if (a.no_kv_offload) {
        ret += "-nkvo";
    }
    for (size_t d = 0; d < a.op_offload_min_batch_dev.size(); d++) {
        const int32_t v = a.op_offload_min_batch_dev[d];
        if (v > 0) {
            ret += "-off" + std::to_string(d) + (v >= FIT_ADVISOR_OFFLOAD_NEVER ? "never" : std::to_string(v));
        }
    }
    return ret;
}

struct searcher {
    const fit_advisor_inventory & inv;
    fit_advisor_probe & probe;
    const std::vector<std::string> & device_bufts;
    const fit_advisor_graph_profile & gp;
    const std::vector<fit_advisor_cost_device> & cost_devs;
    const fit_advisor_pair_table & pairs;
    const fit_advisor_workload & wl_base;
    const fit_advisor_search_options & opts;

    size_t   nd;
    uint32_t n_layer_all;
    uint32_t ngl_max;
    int      n_probes = 0;

    searcher(const fit_advisor_inventory & inv_, fit_advisor_probe & probe_, const std::vector<std::string> & bufts,
             const fit_advisor_graph_profile & gp_, const std::vector<fit_advisor_cost_device> & cd,
             const fit_advisor_pair_table & pr, const fit_advisor_workload & wl, const fit_advisor_search_options & o) :
        inv(inv_), probe(probe_), device_bufts(bufts), gp(gp_), cost_devs(cd), pairs(pr), wl_base(wl), opts(o) {
        nd = device_bufts.size();
        n_layer_all = inv.n_layer + inv.n_layer_nextn;
        ngl_max = n_layer_all + 1;
    }

    // the workload of a state: the base workload at the state's ubatch, drafting as the allocation decides
    fit_advisor_workload workload(uint32_t ub, bool draft_mtp) const {
        fit_advisor_workload wl = wl_base;
        wl.n_ubatch = ub;
        wl.use_mtp  = draft_mtp;
        return wl;
    }
    fit_advisor_workload workload(const fit_advisor_allocation & a) const {
        fit_advisor_workload wl = workload(a.n_ubatch, a.draft_mtp);
        wl.op_offload_min_batch_dev = a.op_offload_min_batch_dev;
        return wl;
    }
    // drafting is explored when the user allowed it (--spec-type draft-mtp) and the model has MTP layers
    bool mtp_searchable() const {
        return wl_base.use_mtp && inv.n_layer_nextn > 0;
    }

    bool tensor_counts(size_t i, const fit_advisor_workload & wl) const {
        return !(inv.tensors[i].layer >= (int32_t) inv.n_layer && !wl.use_mtp);
    }

    // probe an allocation through the loader's own projection, count it
    const fit_advisor_projection & probe_alloc(const fit_advisor_allocation & a, const std::string & name) {
        const fit_advisor_projection & pj = probe.run(a.to_candidate(inv, device_bufts, name));
        n_probes++;
        if (pj.ok) {
            for (size_t d = 0; d < nd && d < pj.devices.size(); d++) {
                const auto & pd = pj.devices[d];
                LOG_DBG("%s: probe %s %s: model %.0f ctx %.0f cmp %.0f scratch %.0f margin %.0f -> left %.0f MiB%s\n", __func__,
                    name.c_str(), pd.name.c_str(), pd.model / (1024.0 * 1024), pd.context / (1024.0 * 1024), pd.compute / (1024.0 * 1024),
                    pd.scratch / (1024.0 * 1024), pd.margin / (1024.0 * 1024), pd.projected_free() / (1024.0 * 1024), pd.fits() ? "" : " OVER");
            }
        }
        return pj;
    }

    // movable groups: the expert tensors of a layer move together (a partially moved layer still costs its split),
    // every other tensor is its own group. only groups with a positive, measured gain are candidates to leave their
    // device; the rest (norms, biases, unmeasured types) always stay with their layer
    using group = fit_advisor_group;

    // a tensor the graph reads at all; everything is a candidate for the annealer, the cost model decides
    bool tensor_used(size_t i) const {
        return i < gp.use_tg.size() && gp.use_tg[i].op != 0;
    }

    // fallback rule: a tensor whose cost is not measured on both its home and the CPU is never moved by the search,
    // it stays with its layer like its neighbours
    bool tensor_decidable(size_t i, int home) const {
        if (i >= gp.use_tg.size()) {
            return false;
        }
        const auto & t = inv.tensors[i];
        return fit_advisor_tensor_cost_known(inv, t, gp.use_tg[i], home, cost_devs)
            && fit_advisor_tensor_cost_known(inv, t, gp.use_tg[i], fit_advisor_allocation::DEV_CPU, cost_devs)
            && fit_advisor_tensor_cost_known(inv, t, gp.use_pp[i], home, cost_devs)
            && fit_advisor_tensor_cost_known(inv, t, gp.use_pp[i], fit_advisor_allocation::DEV_CPU, cost_devs);
    }

    // large enough to seed the knapsack with: the seed works on the tensors that decide memory, the annealer refines
    bool tensor_large(size_t i) const {
        return inv.tensors[i].nbytes >= (size_t) UNIT;
    }

    // the groups of an allocation: a group's home is its layer's device there, and a tensor whose layer sits on the
    // CPU is not a candidate (the leading CPU block is the partition's business)
    std::vector<group> build_groups(const fit_advisor_allocation & base, const fit_advisor_workload & wl) const {
        std::map<int32_t, group> exps_by_layer;
        std::vector<group> ret;
        for (size_t i = 0; i < inv.tensors.size(); i++) {
            const auto & t = inv.tensors[i];
            if (!tensor_counts(i, wl) || t.layer < 0 || !tensor_used(i) || !tensor_large(i)) {
                continue;
            }
            const int home = base.layer_device((uint32_t) t.layer, n_layer_all);
            if (home < 0 || !tensor_decidable(i, home)) {
                continue;
            }
            if (t.kind == FIT_ADVISOR_TENSOR_FFN_EXPS) {
                auto & g = exps_by_layer[t.layer];
                g.idx.push_back(i);
                g.bytes += t.nbytes;
                g.home = home;
            } else {
                ret.push_back({ { i }, t.nbytes, home });
            }
        }
        for (auto & [il, g] : exps_by_layer) {
            ret.push_back(g);
        }
        return ret;
    }

    double score(const fit_advisor_cost & c, const fit_advisor_workload & wl) const {
        return wl.throughput ? -c.gen_tokens_per_s * 1e6 : c.score();
    }
};

} // namespace

// the result from a chosen allocation: the cost with the depth the scan picked, the drafting flag dropped when no
// depth pays, the rate by depth kept for the report, the candidate that reproduces it
static void finish_result(fit_advisor_search_result & best, const fit_advisor_allocation & alloc_in, const fit_advisor_workload & wl,
                          const fit_advisor_projection & proj, const fit_advisor_inventory & inv, const fit_advisor_graph_profile & gp,
                          const std::vector<fit_advisor_cost_device> & cost_devs, const fit_advisor_pair_table & pairs,
                          const std::vector<std::string> & device_bufts, const fit_advisor_workload & wl_base, int n_probes, int accepted) {
    const fit_advisor_allocation alloc = alloc_in;
    best.ok    = true;
    best.alloc = alloc;
    best.wl    = wl;
    best.proj  = proj;
    best.cost  = fit_advisor_cost_estimate(inv, best.alloc, best.proj, gp, cost_devs, pairs, best.wl);
    // the depth the cost model chose is part of what the candidate reproduces; a depth of 0 means drafting never
    // paid on this allocation, which the emitted command then leaves off. the rates by depth stay for the report
    best.alloc.mtp_draft_n = best.cost.mtp_draft_n;
    best.wl.mtp_draft_n    = best.cost.mtp_draft_n;
    if (best.alloc.draft_mtp && best.cost.mtp_draft_n == 0) {
        const std::vector<double> by_depth = best.cost.mtp_depth_tok_s;
        best.alloc.draft_mtp = false;
        best.wl.use_mtp      = false;
        best.cost = fit_advisor_cost_estimate(inv, best.alloc, best.proj, gp, cost_devs, pairs, best.wl);
        best.cost.mtp_depth_tok_s = by_depth;
    }
    if (!best.alloc.draft_mtp && wl_base.use_mtp && inv.n_layer_nextn > 0 && best.cost.mtp_depth_tok_s.empty()) {
        // drafting lost before the walk started: price it on the final allocation anyway so the report can show
        // the rate at each depth (the KV figures of the no-MTP projection stand in for the draft context's)
        fit_advisor_allocation a_mtp = best.alloc;
        a_mtp.draft_mtp = true;
        fit_advisor_workload wl_mtp = best.wl;
        wl_mtp.use_mtp = true;
        const fit_advisor_cost c_mtp = fit_advisor_cost_estimate(inv, a_mtp, best.proj, gp, cost_devs, pairs, wl_mtp);
        if (c_mtp.ok) {
            best.cost.mtp_depth_tok_s = c_mtp.mtp_depth_tok_s;
        }
    }
    best.name  = alloc_name(best.alloc);
    best.cand  = best.alloc.to_candidate(inv, device_bufts, best.name);
    best.n_probes = n_probes;
    best.n_solves = accepted;
}

fit_advisor_search_result fit_advisor_search(const fit_advisor_inventory & inv, fit_advisor_probe & probe,
                                             const std::vector<std::string> & device_bufts,
                                             const fit_advisor_graph_profile & gp,
                                             const std::vector<fit_advisor_cost_device> & cost_devs,
                                             const fit_advisor_pair_table & pairs,
                                             const fit_advisor_workload & wl_base,
                                             const fit_advisor_search_options & opts) {
    fit_advisor_search_result best;
    searcher S(inv, probe, device_bufts, gp, cost_devs, pairs, wl_base, opts);
    const size_t nd = S.nd;
    if (nd == 0) {
        LOG_WRN("%s: no devices, nothing to place\n", __func__);
        return best;
    }

    // ---- the partitions to try: by free memory, even, the fast cards only, the fitter's
    std::vector<std::vector<uint32_t>> partitions;
    {
        const fit_advisor_projection & p0 = probe.run(fit_advisor_allocation::from_layer_split(inv, device_bufts, std::vector<uint32_t>(nd, 0), opts.n_ctx, 1)
                                                      .to_candidate(inv, device_bufts, "search-base"));
        S.n_probes++;
        std::vector<double> w_free(nd, 1.0);
        if (p0.ok) {
            for (size_t d = 0; d < nd; d++) {
                w_free[d] = std::max<double>(1.0, (double) p0.devices[d].free);
            }
        }
        auto split = [&](const std::vector<double> & w) {
            std::vector<uint32_t> per(nd, 0);
            const double sum = std::accumulate(w.begin(), w.end(), 0.0);
            uint32_t assigned = 0;
            size_t last = 0;
            for (size_t d = 0; d < nd; d++) {
                per[d] = (uint32_t) std::floor(S.ngl_max * w[d] / sum);
                assigned += per[d];
                if (w[d] > 0) last = d;
            }
            per[last] += S.ngl_max - assigned; // rounding remainder to the last device that takes layers
            return per;
        };
        auto add_partition = [&](const std::vector<uint32_t> & part) {
            if (std::find(partitions.begin(), partitions.end(), part) == partitions.end()) {
                partitions.push_back(part);
            }
        };
        add_partition(split(w_free));
        add_partition(split(std::vector<double>(nd, 1.0)));
        for (const auto & part : opts.extra_partitions) {
            if (part.size() == nd && std::accumulate(part.begin(), part.end(), 0u) <= S.ngl_max) {
                add_partition(part);
            }
        }

        // unequal devices: every layer's home on the fastest k cards only, the rest hold no layers and get expert
        // stacks from the fill pass instead; a slow card with a slow link is a store, not a place to run attention
        std::vector<size_t> by_rate(nd);
        for (size_t d = 0; d < nd; d++) by_rate[d] = d;
        std::stable_sort(by_rate.begin(), by_rate.end(), [&](size_t a, size_t b) {
            return fit_advisor_device_rate(cost_devs[a]) > fit_advisor_device_rate(cost_devs[b]);
        });
        for (size_t k = nd - 1; k >= 1 && nd > 1; k--) {
            const double rate_k = fit_advisor_device_rate(cost_devs[by_rate[k - 1]]);
            const double rate_x = fit_advisor_device_rate(cost_devs[by_rate[k]]);
            if (rate_x <= 0 || rate_k <= 1.5 * rate_x) {
                continue; // the excluded card is not much slower, the memory-proportional seeds cover it
            }
            std::vector<double> w(nd, 0.0);
            for (size_t i = 0; i < k; i++) w[by_rate[i]] = w_free[by_rate[i]];
            add_partition(split(w));
        }
    }
    std::vector<uint32_t> slot_options;
    for (uint32_t s = 1; s <= std::max<uint32_t>(1, opts.max_slots); s *= 2) {
        slot_options.push_back(s);
    }

    {
        // ---- exact placement: for every key (partition, ubatch, slots, drafting) one probe gives the memory the
        // weights may take, one solve places every movable group and the draft block optimally for the modelled
        // cost, the full cost model prices the solution, the best key wins. a solution that moved the draft block
        // has a new key and is probed again; if the probe finds it over, the capacity comes down and it is re-solved.
        // then the partition is refined around the best key, and the solution polished under the full cost model
        struct solved_key {
            bool ok = false;
            fit_advisor_allocation alloc;
            fit_advisor_projection proj;
            fit_advisor_workload   wl;
            fit_advisor_cost       cost;
            double score = 0;
        };
        int n_solves = 0;
        const uint32_t n_layer_all = S.n_layer_all;
        std::map<bool, std::vector<double>> last_solution; // per drafting choice, the layout is the same across keys

        // ---- the memory model: a dozen probes up front instead of one or two per key. per ubatch and drafting choice
        // three placements are probed: every tensor at home on the memory-proportional split (context per layer,
        // compute of a card that runs layers, scratch), the same with every movable group on the CPU (the copies a
        // card holds for offloaded weights at the prompt batch), and the fast-cards-only split with the empty cards
        // holding expert groups (the compute of a card that is only a store). any placement's overheads follow
        struct overhead_model {
            bool ok = false;
            std::vector<std::string> names;
            std::vector<int64_t> free, margin, scratch, compute_layers, compute_copies, compute_store;
            int64_t ctx_per_layer = 0;
            int64_t host_ctx_per_layer = 0;
            uint32_t n_ctx_train = 0;
        };
        std::map<std::pair<uint32_t, bool>, overhead_model> models;
        auto base_alloc = [&](const std::vector<uint32_t> & part, uint32_t ub, uint32_t slots, bool draft) {
            fit_advisor_allocation base = fit_advisor_allocation::from_layer_split(inv, device_bufts, part, opts.n_ctx, slots);
            base.n_ubatch      = ub;
            base.draft_mtp     = draft;
            base.flash_attn    = opts.base_flash_attn;
            base.no_kv_offload = opts.base_no_kv_offload;
            base.op_offload_min_batch_dev = wl_base.op_offload_min_batch_dev;
            if (base.op_offload_min_batch_dev.empty() && wl_base.op_offload_min_batch > 0) {
                base.op_offload_min_batch_dev.assign(nd, wl_base.op_offload_min_batch);
            }
            return base;
        };
        auto layers_on = [&](const fit_advisor_allocation & a, std::vector<uint32_t> & per, uint32_t & on_cpu) {
            per.assign(nd, 0);
            on_cpu = 0;
            for (uint32_t il = 0; il < n_layer_all; il++) {
                if (il >= inv.n_layer && !a.draft_mtp) continue; // MTP layers not loaded
                const int d = a.layer_device(il, n_layer_all);
                if (d >= 0 && (size_t) d < nd) per[d]++; else on_cpu++;
            }
        };
        auto build_model = [&](uint32_t ub, bool draft) -> overhead_model {
            overhead_model m;
            const uint32_t slots = 1;
            const std::vector<uint32_t> & part = partitions[0];
            fit_advisor_allocation a1 = base_alloc(part, ub, slots, draft);
            const fit_advisor_workload wl1 = S.workload(a1);
            const fit_advisor_projection & p1 = S.probe_alloc(a1, cell_name(part, ub, slots) + (draft ? "-mtp" : "") + "-home");
            if (!p1.ok) return m;
            m.names.resize(nd); m.free.assign(nd, 0); m.margin.assign(nd, 0); m.scratch.assign(nd, 0);
            m.compute_layers.assign(nd, 0); m.compute_copies.assign(nd, 0); m.compute_store.assign(nd, -1);
            m.n_ctx_train = p1.n_ctx_train;
            std::vector<uint32_t> per; uint32_t on_cpu;
            layers_on(a1, per, on_cpu);
            int64_t ctx_sum = 0; uint32_t layers_sum = 0;
            for (size_t d = 0; d < nd && d < p1.devices.size(); d++) {
                m.names[d] = p1.devices[d].name;
                m.free[d] = p1.devices[d].free; m.margin[d] = p1.devices[d].margin;
                m.scratch[d] = (int64_t) p1.devices[d].scratch;
                m.compute_layers[d] = (int64_t) p1.devices[d].compute;
                ctx_sum += (int64_t) p1.devices[d].context; layers_sum += per[d];
            }
            m.ctx_per_layer = layers_sum > 0 ? ctx_sum / layers_sum : 0;
            m.host_ctx_per_layer = on_cpu > 0 ? (int64_t) p1.host.context / on_cpu : m.ctx_per_layer;
            // every movable group on the CPU: the copies the cards hold for them
            fit_advisor_allocation b1 = a1;
            for (const auto & g : S.build_groups(a1, wl1)) {
                for (const size_t i : g.idx) b1.tensor_device[i] = fit_advisor_allocation::DEV_CPU;
            }
            const fit_advisor_projection & pb = S.probe_alloc(b1, cell_name(part, ub, slots) + (draft ? "-mtp" : "") + "-cpu");
            if (pb.ok) {
                for (size_t d = 0; d < nd && d < pb.devices.size(); d++) {
                    m.compute_copies[d] = std::max<int64_t>(0, (int64_t) pb.devices[d].compute - m.compute_layers[d]);
                    m.scratch[d] = std::max<int64_t>(m.scratch[d], (int64_t) pb.devices[d].scratch);
                }
            }
            // a card without layers that holds expert stores: the fast-cards-only split, its empty cards filled
            for (const auto & p2 : partitions) {
                if (std::count(p2.begin(), p2.end(), 0u) == 0 || p2 == part) continue;
                fit_advisor_allocation a2 = base_alloc(p2, ub, slots, draft);
                std::vector<uint32_t> per2; uint32_t cpu2;
                layers_on(a2, per2, cpu2);
                std::vector<int64_t> room(nd, 0);
                for (size_t d = 0; d < nd; d++) room[d] = per2[d] == 0 ? m.free[d] - m.margin[d] - (int64_t) (1024 * UNIT) : 0;
                for (const auto & g : S.build_groups(a2, S.workload(a2))) {
                    if (inv.tensors[g.idx[0]].kind != FIT_ADVISOR_TENSOR_FFN_EXPS && nd > 1) continue;
                    for (size_t d = 0; d < nd; d++) {
                        if (per2[d] == 0 && room[d] >= (int64_t) g.bytes) {
                            for (const size_t i : g.idx) a2.tensor_device[i] = (int) d;
                            room[d] -= (int64_t) g.bytes;
                            break;
                        }
                    }
                }
                const fit_advisor_projection & ps = S.probe_alloc(a2, cell_name(p2, ub, slots) + (draft ? "-mtp" : "") + "-store");
                if (ps.ok) {
                    for (size_t d = 0; d < nd && d < ps.devices.size(); d++) {
                        if (per2[d] == 0) m.compute_store[d] = (int64_t) ps.devices[d].compute + (int64_t) ps.devices[d].scratch - m.scratch[d];
                    }
                }
                break;
            }
            for (size_t d = 0; d < nd; d++) {
                if (m.compute_store[d] < 0) m.compute_store[d] = *std::max_element(m.compute_layers.begin(), m.compute_layers.end());
            }
            m.ok = true;
            return m;
        };
        auto model_for = [&](uint32_t ub, bool draft) -> const overhead_model & {
            auto it = models.find({ ub, draft });
            if (it == models.end()) {
                it = models.emplace(std::make_pair(ub, draft), build_model(ub, draft)).first;
            }
            return it->second;
        };
        // the projection the memory model gives for an allocation: what the cost model and the capacities read
        auto project = [&](const fit_advisor_allocation & a, const overhead_model & m) -> fit_advisor_projection {
            fit_advisor_projection pj;
            pj.ok = m.ok;
            pj.n_ctx_train = m.n_ctx_train;
            std::vector<uint32_t> per; uint32_t on_cpu;
            layers_on(a, per, on_cpu);
            std::vector<size_t> weights(nd, 0);
            bool cpu_offload = false;
            for (size_t i = 0; i < inv.tensors.size() && i < a.tensor_device.size(); i++) {
                const auto & t = inv.tensors[i];
                if (t.layer >= (int32_t) inv.n_layer && !a.draft_mtp) continue;
                const int d = a.tensor_device[i];
                if (d >= 0 && (size_t) d < nd) weights[d] += t.nbytes;
                else if (t.layer >= 0 && a.layer_device((uint32_t) t.layer, n_layer_all) >= 0) cpu_offload = true;
            }
            pj.devices.resize(nd);
            for (size_t d = 0; d < nd; d++) {
                auto & pd = pj.devices[d];
                pd.name    = m.names[d];
                pd.free    = m.free[d];
                pd.margin  = m.margin[d];
                pd.model   = weights[d];
                pd.context = (size_t) (m.ctx_per_layer * per[d]);
                pd.compute = per[d] > 0 ? (size_t) (m.compute_layers[d] + (cpu_offload ? m.compute_copies[d] : 0))
                                        : (weights[d] > 0 ? (size_t) m.compute_store[d] : 0);
                pd.scratch = (size_t) m.scratch[d];
            }
            pj.host.context = (size_t) (m.host_ctx_per_layer * on_cpu);
            return pj;
        };
        auto capacity_of = [&](const fit_advisor_projection & pj) {
            std::vector<int64_t> cap(nd, 0);
            for (size_t d = 0; d < nd && d < pj.devices.size(); d++) {
                const auto & pd = pj.devices[d];
                cap[d] = pd.free - (int64_t) (pd.context + pd.compute + pd.scratch) - pd.margin - (int64_t) (64 * UNIT);
            }
            return cap;
        };

        auto eval_key = [&](const std::vector<uint32_t> & part, uint32_t ub, uint32_t slots, bool draft, const std::vector<int64_t> * cap_override) -> solved_key {
            solved_key out;
            const overhead_model & m = model_for(ub, draft);
            if (!m.ok) {
                return out;
            }
            fit_advisor_allocation base = base_alloc(part, ub, slots, draft);
            const std::string name = cell_name(part, ub, slots) + (draft ? "-mtp" : "");
            best.n_cells++;
            fit_advisor_solve_input in;
            in.inv = &inv; in.gp = &gp; in.devices = &cost_devs; in.pairs = &pairs; in.device_bufts = &device_bufts;
            const fit_advisor_workload wl = S.workload(base);
            in.wl   = &wl;
            in.base = base;
            in.move_draft_block = draft;
            in.groups   = S.build_groups(base, wl);
            in.capacity = cap_override ? *cap_override : capacity_of(project(base, m));
            auto seed_it = last_solution.find(draft);
            if (seed_it != last_solution.end()) {
                in.seed = &seed_it->second;
            }
            fit_advisor_solve_result sol = fit_advisor_solve_placement(in);
            n_solves++;
            if (sol.ok) {
                last_solution[draft] = sol.solution;
            }
            if (!sol.ok) {
                LOG_INF("%s: key %-28s %s\n", __func__, name.c_str(), sol.error.c_str());
                return out;
            }
            out.wl   = S.workload(sol.alloc);
            out.proj = project(sol.alloc, m);
            out.cost = fit_advisor_cost_estimate(inv, sol.alloc, out.proj, gp, cost_devs, pairs, out.wl);
            if (!out.cost.ok) {
                return out;
            }
            out.ok    = true;
            out.alloc = sol.alloc;
            out.score = S.score(out.cost, out.wl);
            LOG_INF("%s: key %-28s gen %6.2f tok/s, pp %6.0f tok/s, request %7.2f s, %d groups, %s in %.1f s\n", __func__, name.c_str(),
                out.cost.gen_tokens_per_s, out.cost.prompt_tokens_per_s, out.cost.t_request_us * 1e-6, sol.n_groups,
                sol.optimal ? "optimal" : "feasible", sol.t_solve_s);
            return out;
        };

        std::vector<bool> draft_options = { false };
        if (S.mtp_searchable()) {
            draft_options = { true, false };
        }
        solved_key best_key;
        std::vector<uint32_t> best_part;
        uint32_t best_ub = 0, best_slots = 1;
        bool best_draft = false;
        for (const auto & part : partitions) {
            for (const uint32_t ub : opts.ubatch_options) {
                for (const uint32_t slots : slot_options) {
                    for (const bool draft : draft_options) {
                        solved_key r = eval_key(part, ub, slots, draft, nullptr);
                        if (r.ok && (!best_key.ok || r.score < best_key.score)) {
                            best_key = r; best_part = part; best_ub = ub; best_slots = slots; best_draft = draft;
                            LOG_INF("%s:   <- best so far\n", __func__);
                        }
                    }
                }
            }
        }
        if (!best_key.ok) {
            LOG_WRN("%s: no key has a feasible placement\n", __func__);
            best.n_probes = S.n_probes;
            return best;
        }

        // ---- partition refinement: the grid's splits are coarse; shift every boundary by up to three layers around
        // the best key, with its ubatch, slots and drafting, until no shift improves it
        if (nd > 1) {
            for (int round = 0; round < 4; round++) {
                bool improved = false;
                std::vector<uint32_t> part0 = best_part;
                for (size_t d = 0; d + 1 < nd && !improved; d++) {
                    for (const int shift : { 1, -1, 2, -2, 3, -3 }) {
                        std::vector<uint32_t> part = part0;
                        if (shift > 0 ? part[d] < (uint32_t) shift : part[d + 1] <= (uint32_t) (-shift)) continue; // the output layer stays with the last device
                        part[d]     = (uint32_t) ((int) part[d] - shift);
                        part[d + 1] = (uint32_t) ((int) part[d + 1] + shift);
                        if (std::find(partitions.begin(), partitions.end(), part) != partitions.end()) continue;
                        partitions.push_back(part); // never twice
                        solved_key r = eval_key(part, best_ub, best_slots, best_draft, nullptr);
                        if (r.ok && r.score < best_key.score) {
                            best_key = r; best_part = part;
                            improved = true;
                            LOG_INF("%s:   <- refined partition\n", __func__);
                            break;
                        }
                    }
                }
                if (!improved) {
                    break;
                }
            }
        }

        // ---- polish: the solver optimised a linear proxy of the cost; a few passes of single-group moves under the
        // full cost model, within the memory the model allows, take what the proxy could not see
        {
            const overhead_model & m = model_for(best_ub, best_draft);
            fit_advisor_allocation cur = best_key.alloc;
            fit_advisor_workload   wl  = best_key.wl;
            double cur_score = best_key.score;
            fit_advisor_cost cur_cost = best_key.cost;
            fit_advisor_projection cur_proj = best_key.proj;
            const std::vector<searcher::group> groups = S.build_groups(best_key.alloc, wl);
            auto fits = [&](const fit_advisor_projection & pj) {
                for (size_t d = 0; d < nd && d < pj.devices.size(); d++) {
                    if (pj.devices[d].projected_free() < pj.devices[d].margin + (int64_t) (64 * UNIT)) return false;
                }
                return true;
            };
            int n_moves = 0;
            for (int pass = 0; pass < 3; pass++) {
                bool any = false;
                for (const auto & g : groups) {
                    const int from = cur.tensor_device[g.idx[0]];
                    for (int to = -1; to < (int) nd; to++) {
                        if (to == from) continue;
                        fit_advisor_allocation nxt = cur;
                        for (const size_t i : g.idx) {
                            nxt.tensor_device[i] = to;
                        }
                        const fit_advisor_projection pj = project(nxt, m);
                        if (!fits(pj)) continue;
                        const fit_advisor_cost c = fit_advisor_cost_estimate(inv, nxt, pj, gp, cost_devs, pairs, wl);
                        if (!c.ok) continue;
                        const double sc = S.score(c, wl);
                        if (sc < cur_score - 1.0) {
                            cur = nxt; cur_score = sc; cur_cost = c; cur_proj = pj; any = true; n_moves++;
                            break;
                        }
                    }
                }
                if (!any) break;
            }
            if (n_moves > 0) {
                LOG_INF("%s: polish moved %d groups under the full cost model: request %.2f -> %.2f s\n", __func__,
                    n_moves, best_key.cost.t_request_us * 1e-6, cur_cost.t_request_us * 1e-6);
                best_key.alloc = cur; best_key.proj = cur_proj; best_key.cost = cur_cost; best_key.score = cur_score;
            }
        }

        // ---- the one real probe: the chosen allocation through the loader's own projection. if a card is over its
        // margin the model missed something there; the key is re-solved with that card's capacity reduced by the
        // shortfall, a few times at most
        {
            std::vector<int64_t> cap = capacity_of(project(base_alloc(best_part, best_ub, best_slots, best_draft), model_for(best_ub, best_draft)));
            for (int round = 0; round < 4; round++) {
                const fit_advisor_projection & pj = S.probe_alloc(best_key.alloc, alloc_name(best_key.alloc));
                if (!pj.ok) {
                    LOG_WRN("%s: the chosen allocation could not be probed\n", __func__);
                    break;
                }
                bool over = false;
                for (size_t d = 0; d < nd && d < pj.devices.size(); d++) {
                    const int64_t deficit = pj.devices[d].margin - pj.devices[d].projected_free();
                    if (deficit > 0) {
                        // the shortfall is in the overheads the model did not see, and the solution may have had slack
                        // under its weight capacity; the capacity has to drop below what the solution placed there,
                        // or the same solution comes back
                        int64_t placed = 0;
                        for (size_t i = 0; i < inv.tensors.size(); i++) {
                            if (best_key.alloc.tensor_device[i] == (int) d) placed += (int64_t) inv.tensors[i].nbytes;
                        }
                        cap[d] = std::min(cap[d], placed) - deficit - (int64_t) (32 * UNIT);
                        over = true;
                        LOG_INF("%s: %s is %.0f MiB over on probe, re-solving with less room there\n", __func__, pj.devices[d].name.c_str(), deficit / (1024.0 * 1024));
                    }
                }
                if (!over) {
                    best_key.proj = pj;
                    best_key.cost = fit_advisor_cost_estimate(inv, best_key.alloc, pj, gp, cost_devs, pairs, best_key.wl);
                    break;
                }
                solved_key r = eval_key(best_part, best_ub, best_slots, best_draft, &cap);
                if (!r.ok) {
                    LOG_WRN("%s: no placement fits the probed capacities, keeping the last one\n", __func__);
                    break;
                }
                best_key = r;
            }
        }

        best.seed_request_us = best_key.cost.t_request_us;
        finish_result(best, best_key.alloc, best_key.wl, best_key.proj, inv, gp, cost_devs, pairs, device_bufts, wl_base, S.n_probes, n_solves);
        return best;
    }
}
