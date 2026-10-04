#include "search.h"
#include "solve.h"

#include "ggml.h"

#include "log.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <map>
#include <numeric>
#include <random>

namespace {

constexpr int64_t UNIT = 1024 * 1024; // memory granularity of the knapsack

// a tensor the search may move between its layer's device and the CPU (or another device)
struct item {
    size_t  idx;
    size_t  bytes;
    bool    mtp_off; // belongs to an MTP layer that is not executed: never counts
};

// per-device request-time contribution of every tensor, for both the CPU and its home, is the same for all cells with
// the same ubatch and slots; cache it
struct gain_table {
    std::vector<double> on_cpu;  // request us if on the CPU
    std::vector<double> on_dev;  // request us if on a device (rates are per device, recomputed per device index below)
};

struct memory_model {
    // per device: bytes of context + compute + scratch from the last probe of a (partition, ubatch, slots) key,
    // plus the free and margin figures; weights are additive on top
    struct entry {
        std::vector<int64_t> overhead;
        std::vector<int64_t> free;
        std::vector<int64_t> margin;
        bool ok = false;
    };
    std::map<std::string, entry> by_key;
};

std::string cell_key(const std::vector<uint32_t> & part, uint32_t ub, uint32_t slots) {
    std::string k;
    for (uint32_t x : part) {
        k += std::to_string(x) + "/";
    }
    return k + "ub" + std::to_string(ub) + "/np" + std::to_string(slots);
}

// the key of a state: its cell plus any explicit layer homes, which change the KV and compute of the devices
std::string alloc_key(const fit_advisor_allocation & a) {
    std::string ret = cell_key(a.layers_per_device, a.n_ubatch, a.n_slots);
    for (const auto & [il, dev] : a.layer_home) {
        ret += "|" + std::to_string(il) + "=" + std::to_string(dev);
    }
    if (a.draft_mtp) {
        ret += "|mtp";
    }
    if (a.flash_attn >= 0) {
        ret += a.flash_attn ? "|fa" : "|nofa";
    }
    if (a.no_kv_offload) {
        ret += "|nkvo";
    }
    for (size_t d = 0; d < a.op_offload_min_batch_dev.size(); d++) {
        if (a.op_offload_min_batch_dev[d] > 0) {
            ret += "|off" + std::to_string(d) + "=" + std::to_string(a.op_offload_min_batch_dev[d]);
        }
    }
    return ret;
}

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
    memory_model mem;

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

    // probe an allocation, record its overheads for its key, count it
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
            memory_model::entry e;
            e.ok = true;
            for (size_t d = 0; d < nd && d < pj.devices.size(); d++) {
                e.overhead.push_back((int64_t) (pj.devices[d].context + pj.devices[d].compute + pj.devices[d].scratch));
                e.free.push_back(pj.devices[d].free);
                e.margin.push_back(pj.devices[d].margin);
            }
            mem.by_key[alloc_key(a)] = e;
        }
        return pj;
    }

    // linear memory model: weights on each device plus the probed overhead for the key; returns the over-budget bytes
    // per device (negative = room), or false if the key was never probed
    bool memory_over(const fit_advisor_allocation & a, const fit_advisor_workload & wl, std::vector<int64_t> & over) const {
        const auto it = mem.by_key.find(alloc_key(a));
        if (it == mem.by_key.end() || !it->second.ok) {
            return false;
        }
        over.assign(nd, 0);
        for (size_t i = 0; i < inv.tensors.size(); i++) {
            const int d = a.tensor_device[i];
            if (d >= 0 && (size_t) d < nd && tensor_counts(i, wl)) {
                over[d] += inv.tensors[i].nbytes;
            }
        }
        for (size_t d = 0; d < nd; d++) {
            over[d] = over[d] + it->second.overhead[d] - (it->second.free[d] - it->second.margin[d]);
        }
        return true;
    }

    // request-time gain of placing tensor i on device d instead of the CPU, from the op that reads it
    double gain(size_t i, int d, const fit_advisor_workload & wl, uint32_t slots) const {
        return fit_advisor_tensor_request_us(inv, gp, i, fit_advisor_allocation::DEV_CPU, cost_devs, wl, slots, d)
             - fit_advisor_tensor_request_us(inv, gp, i, d, cost_devs, wl, slots, d);
    }

    // movable groups: the expert tensors of a layer move together (a partially moved layer still costs its split),
    // every other tensor is its own group. only groups with a positive, measured gain are candidates to leave their
    // device; the rest (norms, biases, unmeasured types) always stay with their layer
    struct group {
        std::vector<size_t> idx;
        size_t bytes = 0;
    };

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

    std::vector<group> build_groups(const fit_advisor_workload & wl) const {
        std::map<int32_t, group> exps_by_layer;
        std::vector<group> ret;
        for (size_t i = 0; i < inv.tensors.size(); i++) {
            const auto & t = inv.tensors[i];
            if (!tensor_counts(i, wl) || t.layer < 0 || !tensor_used(i) || !tensor_large(i)) {
                continue;
            }
            const int home = fit_advisor_allocation::from_layer_split(inv, device_bufts, std::vector<uint32_t>(nd, 1), 0, 1).tensor_device[i];
            if (home >= 0 && !tensor_decidable(i, home)) {
                continue;
            }
            if (t.kind == FIT_ADVISOR_TENSOR_FFN_EXPS) {
                exps_by_layer[t.layer].idx.push_back(i);
                exps_by_layer[t.layer].bytes += t.nbytes;
            } else {
                ret.push_back({ { i }, t.nbytes });
            }
        }
        for (auto & [il, g] : exps_by_layer) {
            ret.push_back(g);
        }
        return ret;
    }

    double group_gain(const group & g, int d, const fit_advisor_workload & wl, uint32_t slots) const {
        double ret = 0;
        for (const size_t i : g.idx) {
            ret += gain(i, d, wl, slots);
        }
        return ret;
    }

    // exact 0/1 knapsack per device over the movable groups of its layers: which to keep on it under a byte capacity
    // everything that is not a movable group stays where the layer split put it and counts against the capacity
    void knapsack_fill(fit_advisor_allocation & a, const fit_advisor_allocation & base, const std::vector<int64_t> & capacity,
                       const fit_advisor_workload & wl, const std::vector<group> & groups) const {
        a = base;
        for (size_t d = 0; d < nd; d++) {
            std::vector<size_t> gidx;
            std::vector<int64_t> w;
            std::vector<double> v;
            int64_t fixed = 0;
            std::vector<char> is_movable(inv.tensors.size(), 0);
            for (size_t k = 0; k < groups.size(); k++) {
                const group & g = groups[k];
                if (base.tensor_device[g.idx[0]] != (int) d) {
                    continue;
                }
                const double gn = group_gain(g, (int) d, wl, a.n_slots);
                if (gn <= 0) {
                    continue; // stays home
                }
                for (const size_t i : g.idx) {
                    is_movable[i] = 1;
                }
                gidx.push_back(k);
                w.push_back((int64_t) ((g.bytes + UNIT - 1) / UNIT));
                v.push_back(gn);
            }
            for (size_t i = 0; i < inv.tensors.size(); i++) {
                if (base.tensor_device[i] == (int) d && !is_movable[i] && tensor_counts(i, wl)) {
                    fixed += inv.tensors[i].nbytes;
                }
            }
            const int64_t cap = std::max<int64_t>(0, (capacity[d] - fixed) / UNIT);
            const size_t n = gidx.size();
            // default: movable groups off the device, the knapsack puts the chosen ones back
            for (const size_t k : gidx) {
                for (const size_t i : groups[k].idx) {
                    a.tensor_device[i] = fit_advisor_allocation::DEV_CPU;
                }
            }
            if (n == 0 || cap == 0) {
                continue;
            }
            std::vector<double> best(cap + 1, 0.0);
            std::vector<uint8_t> keep(n * (cap + 1), 0);
            for (size_t k = 0; k < n; k++) {
                const int64_t wk = w[k];
                if (wk > cap) {
                    continue;
                }
                for (int64_t c = cap; c >= wk; c--) {
                    const double cand = best[c - wk] + v[k];
                    if (cand > best[c]) {
                        best[c] = cand;
                        keep[k * (cap + 1) + c] = 1;
                    }
                }
            }
            int64_t c = cap;
            for (size_t k = n; k-- > 0;) {
                if (keep[k * (cap + 1) + c]) {
                    for (const size_t i : groups[gidx[k]].idx) {
                        a.tensor_device[i] = (int) d;
                    }
                    c -= w[k];
                }
            }
        }
    }

    // seed one cell: knapsack against probe-corrected capacities until the probe agrees
    struct cell {
        fit_advisor_allocation alloc;
        fit_advisor_projection proj;
        fit_advisor_cost cost;
        bool fits = false;
    };

    cell solve_cell(const std::vector<uint32_t> & part, uint32_t ub, uint32_t slots) {
        cell r;
        const fit_advisor_workload wl = workload(ub, wl_base.use_mtp);
        const std::string name = cell_name(part, ub, slots);

        fit_advisor_allocation base = fit_advisor_allocation::from_layer_split(inv, device_bufts, part, opts.n_ctx, slots);
        base.draft_mtp     = wl_base.use_mtp;
        base.flash_attn    = opts.base_flash_attn;
        base.no_kv_offload = opts.base_no_kv_offload;
        base.op_offload_min_batch_dev = wl_base.op_offload_min_batch_dev;
        if (base.op_offload_min_batch_dev.empty() && wl_base.op_offload_min_batch > 0) {
            base.op_offload_min_batch_dev.assign(nd, wl_base.op_offload_min_batch);
        }
        base.n_ubatch = ub;
        const std::vector<group> groups = build_groups(wl);

        // floor: the expert groups on the CPU, dense weights stay with their layers; only if that does not fit are the
        // dense groups moved off too. (with everything on the CPU and a large batch every op is offloaded and the compute
        // buffer holds far more weight copies at once, so that floor can fail where a real allocation would not)
        auto make_floor = [&](bool experts_only) {
            fit_advisor_allocation f = base;
            for (const group & g : groups) {
                const int home = base.tensor_device[g.idx[0]];
                if (home < 0 || group_gain(g, home, wl, slots) <= 0) {
                    continue;
                }
                if (experts_only && inv.tensors[g.idx[0]].kind != FIT_ADVISOR_TENSOR_FFN_EXPS) {
                    continue;
                }
                for (const size_t i : g.idx) {
                    f.tensor_device[i] = fit_advisor_allocation::DEV_CPU;
                }
            }
            return f;
        };
        fit_advisor_allocation floor = make_floor(true);
        const fit_advisor_projection * pj = &probe_alloc(floor, name);
        if (pj->ok && !pj->fits_all()) {
            floor = make_floor(false);
            pj = &probe_alloc(floor, name);
        }
        if (!pj->ok) {
            r.proj = *pj;
            return r;
        }

        fit_advisor_allocation a = floor;
        for (int iter = 0; iter < 5; iter++) {
            std::vector<int64_t> cap(nd, 0);
            for (size_t d = 0; d < nd; d++) {
                // room the probe reports plus what this allocation already holds
                int64_t held = 0;
                for (size_t i = 0; i < inv.tensors.size(); i++) {
                    if (a.tensor_device[i] == (int) d && tensor_counts(i, wl)) {
                        held += inv.tensors[i].nbytes;
                    }
                }
                cap[d] = held + (pj->devices[d].projected_free() - pj->devices[d].margin);
            }
            fit_advisor_allocation next;
            knapsack_fill(next, base, cap, wl, groups);
            next.n_ubatch = ub;
            if (next.tensor_device == a.tensor_device) {
                break;
            }
            a = next;
            pj = &probe_alloc(a, name);
            if (!pj->ok) {
                break;
            }
            if (pj->fits_all()) {
                bool tight = true;
                for (size_t d = 0; d < nd; d++) {
                    if (pj->devices[d].projected_free() - pj->devices[d].margin > (int64_t) (256 * UNIT)) {
                        tight = false;
                    }
                }
                if (tight) {
                    break;
                }
            }
        }
        r.alloc = a;
        r.proj  = *pj;
        r.fits  = pj->ok && pj->fits_all();
        if (pj->ok) {
            r.cost = fit_advisor_cost_estimate(inv, a, *pj, gp, cost_devs, pairs, wl);
            if (a.draft_mtp && r.cost.ok && r.cost.mtp_draft_n == 0) {
                // drafting does not pay here: the MTP layers would only cost memory, so the seed starts without them
                a.draft_mtp = false;
                pj = &probe_alloc(a, name);
                r.alloc = a;
                r.proj  = *pj;
                r.fits  = pj->ok && pj->fits_all();
                if (pj->ok) {
                    r.cost = fit_advisor_cost_estimate(inv, a, *pj, gp, cost_devs, pairs, workload(a));
                }
            }
            r.alloc.mtp_draft_n = r.cost.mtp_draft_n;
        }
        return r;
    }

    double score(const fit_advisor_cost & c, const fit_advisor_workload & wl) const {
        return wl.throughput ? -c.gen_tokens_per_s * 1e6 : c.score();
    }

    // ---- annealing over the full state

    struct state {
        fit_advisor_allocation alloc;
        fit_advisor_workload   wl;
        double objective = 0; // cost score plus memory penalty
        double cost_score = 0;
        double penalty = 0;
    };

    // the projection used for the cost of a state: the probe of its key with this allocation's own KV/compute figures
    // (context/compute/scratch depend on the key, the weights term is not used by the cost model)
    bool evaluate(state & s, const fit_advisor_projection & proj_for_key) {
        std::vector<int64_t> over;
        if (!memory_over(s.alloc, s.wl, over)) {
            return false;
        }
        const fit_advisor_cost c = fit_advisor_cost_estimate(inv, s.alloc, proj_for_key, gp, cost_devs, pairs, s.wl);
        if (!c.ok) {
            return false;
        }
        if (s.alloc.draft_mtp && c.mtp_draft_n == 0) {
            // the MTP layers loaded but no depth pays: the same allocation without them is as fast and holds less,
            // so this state is never entered (the drafting toggle leads there instead)
            return false;
        }
        s.alloc.mtp_draft_n = c.mtp_draft_n;
        s.wl.mtp_draft_n    = c.mtp_draft_n;
        s.cost_score = score(c, s.wl);
        s.penalty = 0;
        for (size_t d = 0; d < nd; d++) {
            if (over[d] > 0) {
                // over budget: not a state the walk may enter. swaps and whole-group moves still let it rearrange
                s.penalty += over[d] / (double) UNIT * 2000.0;
            }
        }
        s.objective = s.cost_score + s.penalty;
        return s.penalty == 0;
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
    best.n_anneal_accepted = accepted;
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

    // ---- 1. seeds: exact knapsack per grid cell
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

#ifdef FIT_ADVISOR_HIGHS
    if (fit_advisor_solver_available() && !opts.anneal) {
        // ---- exact placement: for every key (partition, ubatch, slots, drafting) one probe gives the memory the
        // weights may take, one solve places every movable group and the draft block optimally for the modelled
        // cost, the full cost model prices the solution, the best key wins. a solution that moved the draft block
        // has a new key and is probed again; if the probe finds it over, the capacity comes down and it is re-solved
        fit_advisor_allocation best_alloc;
        fit_advisor_projection best_proj;
        fit_advisor_workload   best_wl;
        fit_advisor_cost       best_cost;
        double best_score = 0;
        bool   have_best  = false;
        std::vector<bool> draft_options = { false };
        if (S.mtp_searchable()) {
            draft_options = { true, false };
        }
        int n_solves = 0;
        for (const auto & part : partitions) {
            for (const uint32_t ub : opts.ubatch_options) {
                for (const uint32_t slots : slot_options) {
                    for (const bool draft : draft_options) {
                        fit_advisor_allocation base = fit_advisor_allocation::from_layer_split(inv, device_bufts, part, opts.n_ctx, slots);
                        base.n_ubatch      = ub;
                        base.draft_mtp     = draft;
                        base.flash_attn    = opts.base_flash_attn;
                        base.no_kv_offload = opts.base_no_kv_offload;
                        base.op_offload_min_batch_dev = wl_base.op_offload_min_batch_dev;
                        if (base.op_offload_min_batch_dev.empty() && wl_base.op_offload_min_batch > 0) {
                            base.op_offload_min_batch_dev.assign(nd, wl_base.op_offload_min_batch);
                        }
                        const std::string name = cell_name(part, ub, slots) + (draft ? "-mtp" : "");
                        const fit_advisor_projection * pj = &S.probe_alloc(base, name);
                        best.n_cells++;
                        if (!pj->ok) {
                            LOG_INF("%s: key %-28s probe failed\n", __func__, name.c_str());
                            continue;
                        }
                        fit_advisor_solve_input in;
                        in.inv = &inv; in.gp = &gp; in.devices = &cost_devs; in.pairs = &pairs; in.device_bufts = &device_bufts;
                        const fit_advisor_workload wl = S.workload(base);
                        in.wl   = &wl;
                        in.base = base;
                        in.move_draft_block = draft;
                        in.capacity.assign(nd, 0);
                        for (size_t d = 0; d < nd && d < pj->devices.size(); d++) {
                            const auto & pd = pj->devices[d];
                            in.capacity[d] = pd.free - (int64_t) (pd.context + pd.compute + pd.scratch) - pd.margin - (int64_t) (64 * UNIT);
                        }
                        fit_advisor_solve_result sol;
                        for (int round = 0; round < 3; round++) {
                            sol = fit_advisor_solve_placement(in);
                            n_solves++;
                            if (!sol.ok) {
                                break;
                            }
                            // every solution is checked by the loader's own projection; the draft block's move
                            // changes the key, and a packed card can read a few MiB differently
                            const fit_advisor_projection & pj2 = S.probe_alloc(sol.alloc, name);
                            if (!pj2.ok) {
                                sol.ok = false;
                                sol.error = "probe of the solution failed";
                                break;
                            }
                            bool over = false;
                            for (size_t d = 0; d < nd && d < pj2.devices.size(); d++) {
                                const int64_t deficit = pj2.devices[d].margin - pj2.devices[d].projected_free();
                                if (deficit > 0) {
                                    in.capacity[d] -= deficit + (int64_t) (32 * UNIT);
                                    over = true;
                                }
                            }
                            pj = &pj2;
                            if (!over) {
                                break;
                            }
                            if (round == 2) {
                                sol.ok = false;
                                sol.error = "the solution did not fit on probe after three rounds";
                            }
                        }
                        if (!sol.ok) {
                            LOG_INF("%s: key %-28s %s\n", __func__, name.c_str(), sol.error.c_str());
                            continue;
                        }
                        const fit_advisor_workload wl_sol = S.workload(sol.alloc);
                        const fit_advisor_cost cost = fit_advisor_cost_estimate(inv, sol.alloc, *pj, gp, cost_devs, pairs, wl_sol);
                        if (!cost.ok) {
                            continue;
                        }
                        const double score = S.score(cost, wl_sol);
                        const bool better = !have_best || score < best_score;
                        LOG_INF("%s: key %-28s gen %6.2f tok/s, pp %6.0f tok/s, request %7.2f s, %d groups, %s in %.1f s%s\n", __func__, name.c_str(),
                            cost.gen_tokens_per_s, cost.prompt_tokens_per_s, cost.t_request_us * 1e-6, sol.n_groups,
                            sol.optimal ? "optimal" : "feasible", sol.t_solve_s, better ? "  <- best" : "");
                        if (better) {
                            have_best  = true;
                            best_score = score;
                            best_alloc = sol.alloc;
                            best_proj  = *pj;
                            best_wl    = wl_sol;
                            best_cost  = cost;
                        }
                    }
                }
            }
        }
        if (!have_best) {
            LOG_WRN("%s: no key has a feasible placement\n", __func__);
            best.n_probes = S.n_probes;
            return best;
        }
        best.seed_request_us = best_cost.t_request_us;
        finish_result(best, best_alloc, best_wl, best_proj, inv, gp, cost_devs, pairs, device_bufts, wl_base, S.n_probes, n_solves);
        return best;
    }
#endif

    searcher::cell best_cell;
    std::vector<uint32_t> best_part;
    uint32_t best_ub = 0;
    if (opts.has_warm_start) {
        // the previous result as the only seed: probed and priced under the current costs
        searcher::cell r;
        r.alloc = opts.warm_start;
        r.proj  = S.probe_alloc(r.alloc, alloc_name(r.alloc));
        r.fits  = r.proj.ok && r.proj.fits_all();
        if (r.proj.ok) {
            r.cost = fit_advisor_cost_estimate(inv, r.alloc, r.proj, gp, cost_devs, pairs, S.workload(r.alloc));
        }
        if (r.fits && r.cost.ok) {
            LOG_INF("%s: warm start from %s: gen %6.2f tok/s, pp %6.0f tok/s, request %7.2f s\n", __func__, alloc_name(r.alloc).c_str(),
                r.cost.gen_tokens_per_s, r.cost.prompt_tokens_per_s, r.cost.t_request_us * 1e-6);
            best_cell = r;
            best_part = r.alloc.layers_per_device;
            best_ub   = r.alloc.n_ubatch;
            best.n_cells++;
        } else {
            LOG_WRN("%s: the warm start does not fit under the current costs, seeding from the grid\n", __func__);
        }
    }
    for (const auto & part : partitions) {
        if (best_cell.fits && opts.has_warm_start) {
            break;
        }
        for (const uint32_t ub : opts.ubatch_options) {
            for (const uint32_t slots : slot_options) {
                const std::string name = cell_name(part, ub, slots);
                searcher::cell r = S.solve_cell(part, ub, slots);
                best.n_cells++;
                if (!r.fits || !r.cost.ok) {
                    LOG_INF("%s: seed %-28s does not fit\n", __func__, name.c_str());
                    continue;
                }
                const fit_advisor_workload wl = S.workload(r.alloc);
                const bool better = !best_cell.fits || S.score(r.cost, wl) < S.score(best_cell.cost, S.workload(best_cell.alloc));
                LOG_INF("%s: seed %-28s gen %6.2f tok/s, pp %6.0f tok/s, request %7.2f s%s\n", __func__, name.c_str(),
                    r.cost.gen_tokens_per_s, r.cost.prompt_tokens_per_s, r.cost.t_request_us * 1e-6, better ? "  <- best seed" : "");
                if (better) {
                    best_cell = r;
                    best_part = part;
                    best_ub = ub;
                }
            }
        }
    }
    if (!best_cell.fits) {
        LOG_WRN("%s: no seed fits\n", __func__);
        best.n_probes = S.n_probes;
        return best;
    }
    best.seed_request_us = best_cell.cost.t_request_us;

    // ---- 2. simulated annealing from the best seed
    std::mt19937 rng(opts.seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    searcher::state cur;
    cur.alloc = best_cell.alloc;
    cur.wl    = S.workload(cur.alloc);
    std::map<std::string, fit_advisor_projection> proj_by_key; // last probe per key, for the cost model's KV figures
    proj_by_key[alloc_key(cur.alloc)] = best_cell.proj;
    if (!S.evaluate(cur, best_cell.proj)) {
        LOG_WRN("%s: cannot evaluate the seed\n", __func__);
        best.n_probes = S.n_probes;
        return best;
    }
    searcher::state incumbent = cur;
    fit_advisor_projection incumbent_proj = best_cell.proj;
    searcher::state best_seen = cur;
    bool best_seen_dirty = false;
    {
        // what the linear memory model holds for the seed's key, against the seed's own projection
        const std::string key0 = alloc_key(cur.alloc);
        const auto it = S.mem.by_key.find(key0);
        std::vector<int64_t> over;
        S.memory_over(cur.alloc, cur.wl, over);
        for (size_t d = 0; d < nd && it != S.mem.by_key.end() && d < best_cell.proj.devices.size(); d++) {
            const auto & pd = best_cell.proj.devices[d];
            LOG_DBG("%s: seed memory model %s: overhead %lld MiB (probe ctx+cmp+scratch %lld), free %lld, margin %lld -> over %lld MiB; probe model %lld, left %lld\n", __func__,
                device_bufts[d].c_str(), (long long) (it->second.overhead[d] >> 20), (long long) ((pd.context + pd.compute + pd.scratch) >> 20),
                (long long) (it->second.free[d] >> 20), (long long) (it->second.margin[d] >> 20), (long long) (over[d] >> 20),
                (long long) (pd.model >> 20), (long long) (pd.projected_free() >> 20));
        }
        LOG_DBG("%s: seed penalty %.4f s\n", __func__, cur.penalty * 1e-6);
    }

    // movable groups: those with a positive gain on their home; plus every used layer tensor as a single candidate
    std::vector<searcher::group> movable;
    std::vector<size_t> singles;
    {
        fit_advisor_allocation home = fit_advisor_allocation::from_layer_split(inv, device_bufts, cur.alloc.layers_per_device, opts.n_ctx, cur.alloc.n_slots, cur.alloc.layer_home);
        for (const auto & g : S.build_groups(cur.wl)) {
            const int h = home.tensor_device[g.idx[0]];
            if (h >= 0 && S.group_gain(g, h, cur.wl, cur.alloc.n_slots) > 0) {
                movable.push_back(g);
            }
        }
        for (size_t i = 0; i < inv.tensors.size(); i++) {
            if (inv.tensors[i].layer >= 0 && home.tensor_device[i] >= 0 && S.tensor_counts(i, cur.wl) && S.tensor_used(i)
                && S.tensor_decidable(i, home.tensor_device[i])) {
                singles.push_back(i);
            }
        }
    }
    if (movable.empty()) {
        LOG_INF("%s: nothing movable, keeping the seed\n", __func__);
    }
    const char * trace_tensor = getenv("FIT_ADVISOR_TRACE"); // substring of a tensor name whose moves are logged
    auto set_group = [&](fit_advisor_allocation & a, const searcher::group & g, int dev) {
        for (const size_t i : g.idx) {
            a.tensor_device[i] = dev;
        }
    };
    // a destination for a move: the home device, any other device, or the CPU, never where the tensor already is.
    // the home and the CPU get half the draws between them, other devices share the rest (an expert store on a
    // third card is exactly the arrangement worth finding on unequal hardware)
    auto pick_target = [&](int current, int home_dev) -> int {
        std::vector<int> options;
        if (current != home_dev) options.push_back(home_dev);
        if (current != fit_advisor_allocation::DEV_CPU) options.push_back(fit_advisor_allocation::DEV_CPU);
        std::vector<int> others;
        for (int d = 0; d < (int) nd; d++) {
            if (d != current && d != home_dev) others.push_back(d);
        }
        if (others.empty() || (!options.empty() && uni(rng) < 0.5)) {
            return options.empty() ? current : options[(size_t) (uni(rng) * options.size()) % options.size()];
        }
        return others[(size_t) (uni(rng) * others.size()) % others.size()];
    };
    auto cost_known_on = [&](size_t i, int d) {
        return i < gp.use_tg.size()
            && fit_advisor_tensor_cost_known(inv, inv.tensors[i], gp.use_tg[i], d, cost_devs)
            && fit_advisor_tensor_cost_known(inv, inv.tensors[i], gp.use_pp[i], d, cost_devs);
    };

    // fill pass: from a state, try every CPU-resident movable group on every device and every CPU-resident single on
    // its home, best weight-only gain per byte first, keeping each move the full model accepts within the memory
    // model. run before annealing so an expert store on a spare card is in the seed, and again at the end
    auto fill_pass = [&](searcher::state & st, std::map<std::string, fit_advisor_projection> & pbk, const char * when) {
        fit_advisor_allocation home = fit_advisor_allocation::from_layer_split(inv, device_bufts, st.alloc.layers_per_device, opts.n_ctx, st.alloc.n_slots, st.alloc.layer_home);
        struct cand { int group; size_t single; int dev; double density; };
        std::vector<cand> cands;
        for (size_t k = 0; k < movable.size(); k++) {
            const searcher::group & g = movable[k];
            if (st.alloc.tensor_device[g.idx[0]] != fit_advisor_allocation::DEV_CPU) continue;
            for (int d = 0; d < (int) nd; d++) {
                double gsum = 0;
                bool known = true;
                for (const size_t i : g.idx) {
                    known = known && cost_known_on(i, d);
                    gsum += S.gain(i, d, st.wl, st.alloc.n_slots);
                }
                if (known) cands.push_back({ (int) k, 0, d, gsum / (double) g.bytes });
            }
        }
        for (const size_t i : singles) {
            if (st.alloc.tensor_device[i] != fit_advisor_allocation::DEV_CPU || home.tensor_device[i] < 0) continue;
            cands.push_back({ -1, i, home.tensor_device[i], S.gain(i, home.tensor_device[i], st.wl, st.alloc.n_slots) / (double) inv.tensors[i].nbytes });
        }
        std::sort(cands.begin(), cands.end(), [](const cand & a, const cand & b) { return a.density > b.density; });
        int n_filled = 0;
        const std::string fkey = alloc_key(st.alloc);
        if (!pbk.count(fkey)) return 0;
        for (const cand & c : cands) {
            searcher::state nxt = st;
            if (c.group >= 0) {
                if (st.alloc.tensor_device[movable[c.group].idx[0]] != fit_advisor_allocation::DEV_CPU) continue; // taken by an earlier candidate
                set_group(nxt.alloc, movable[c.group], c.dev);
            } else {
                if (st.alloc.tensor_device[c.single] != fit_advisor_allocation::DEV_CPU) continue;
                nxt.alloc.tensor_device[c.single] = c.dev;
            }
            bool feasible = S.evaluate(nxt, pbk[fkey]);
            if (feasible) {
                // a packed device must keep a little below its margin: the probe of the packed state reads a few MiB
                // differently from the linear model, and a state that only just fits would be thrown out for it
                std::vector<int64_t> over;
                if (S.memory_over(nxt.alloc, nxt.wl, over)) {
                    for (size_t d = 0; d < over.size(); d++) {
                        feasible = feasible && over[d] + (int64_t) (64 * UNIT) <= 0;
                    }
                }
            }
            if (trace_tensor) {
                const size_t i = c.group >= 0 ? movable[c.group].idx[0] : c.single;
                if (inv.tensors[i].name.find(trace_tensor) != std::string::npos) {
                    LOG_INF("%s: fill %-34s CPU -> %s: %s, cost %.4f -> %.4f s\n", __func__, inv.tensors[i].name.c_str(),
                        device_bufts[c.dev].c_str(), feasible ? "fits" : "over budget", st.cost_score * 1e-6, nxt.cost_score * 1e-6);
                }
            }
            if (feasible && nxt.objective < st.objective) {
                st = nxt;
                n_filled++;
            }
        }
        if (n_filled > 0) {
            LOG_INF("%s: fill pass %s placed %d more groups or tensors, model now %.3f s\n", __func__, when, n_filled, st.objective * 1e-6);
        }
        return n_filled;
    };

    int n_improvements = 0;
    int n_single_tried = 0, n_single_accepted = 0, n_rehome_tried = 0, n_mtp_tried = 0, n_offload_tried = 0, n_attn_tried = 0, n_exchange_tried = 0;
    const double T0 = std::max(1.0, 0.005 * std::fabs(cur.cost_score));
    const double T1 = std::max(0.01, 0.00002 * std::fabs(cur.cost_score));
    const int iters = std::max(0, opts.anneal_iters);
    int accepted = 0;
    int last_probe_iter = 0;

    // spare capacity anywhere, e.g. a smaller third card with no layers of its own, is filled with the best CPU-resident
    // groups before the walk starts, so the annealer refines that arrangement instead of having to discover it.
    // the fill trusts the memory model of the seed's key, and that model knows nothing about a device the seed left
    // empty: holding expert groups costs a compute buffer and scratch there. so the packed state is probed at once; the
    // probe corrects the model for the key and the fill is redone from the seed until the packed state fits
    {
        const searcher::state seed = cur;
        for (int round = 0; round < 3; round++) {
            searcher::state st = seed;
            const std::string k = alloc_key(st.alloc);
            if (!S.evaluate(st, proj_by_key[k]) || st.penalty > 0) {
                break;
            }
            if (fill_pass(st, proj_by_key, "before annealing") == 0) {
                break;
            }
            const fit_advisor_projection & pj = S.probe_alloc(st.alloc, alloc_name(st.alloc));
            proj_by_key[k] = pj;
            searcher::state checked = st;
            if (pj.ok && pj.fits_all() && S.evaluate(checked, pj) && checked.penalty == 0) {
                cur = checked;
                if (cur.objective < incumbent.objective) {
                    incumbent      = cur; // verified by the loader, so it can stand as the incumbent right away
                    incumbent_proj = pj;
                    best_seen      = cur;
                }
                break;
            }
            LOG_INF("%s: the filled seed does not fit on probe (round %d), refilling with the probed overheads\n", __func__, round + 1);
            cur = seed;
            S.evaluate(cur, proj_by_key[k]);
        }
    }

    for (int it = 0; it < iters && (!movable.empty() || !singles.empty()); it++) {
        const double T = T0 * std::pow(T1 / T0, (double) it / std::max(1, iters));
        searcher::state nxt = cur;
        const double mv = uni(rng);

        if (mv < 0.15) {
            // toggle a single tensor of any kind: partial expert layers, a stray norm, whatever the model prices as better
            if (singles.empty()) continue;
            // draw large tensors more often, they decide memory; small ones still get their turn
            size_t i = singles[(size_t) (uni(rng) * singles.size()) % singles.size()];
            if (inv.tensors[i].nbytes < (size_t) UNIT && uni(rng) < 0.8) {
                i = singles[(size_t) (uni(rng) * singles.size()) % singles.size()];
            }
            fit_advisor_allocation home = fit_advisor_allocation::from_layer_split(inv, device_bufts, nxt.alloc.layers_per_device, opts.n_ctx, nxt.alloc.n_slots, nxt.alloc.layer_home);
            const int target = pick_target(nxt.alloc.tensor_device[i], home.tensor_device[i]);
            if (target == nxt.alloc.tensor_device[i] || (target >= 0 && !cost_known_on(i, target))) continue;
            nxt.alloc.tensor_device[i] = target;
            n_single_tried++;
        } else if (mv < 0.55) {
            // toggle one group between its home and the CPU
            const searcher::group & g = movable[(size_t) (uni(rng) * movable.size()) % movable.size()];
            fit_advisor_allocation home = fit_advisor_allocation::from_layer_split(inv, device_bufts, nxt.alloc.layers_per_device, opts.n_ctx, nxt.alloc.n_slots, nxt.alloc.layer_home);
            const int h = home.tensor_device[g.idx[0]];
            const int target = pick_target(nxt.alloc.tensor_device[g.idx[0]], h);
            if (target == nxt.alloc.tensor_device[g.idx[0]]) continue;
            bool known = true;
            for (const size_t i : g.idx) known = known && (target < 0 || cost_known_on(i, target));
            if (!known) continue;
            set_group(nxt.alloc, g, target);
        } else if (mv < 0.85) {
            // swap one on-device group with one CPU group of the same home
            std::vector<size_t> on, off;
            fit_advisor_allocation home = fit_advisor_allocation::from_layer_split(inv, device_bufts, nxt.alloc.layers_per_device, opts.n_ctx, nxt.alloc.n_slots, nxt.alloc.layer_home);
            const int d = (int) ((size_t) (uni(rng) * nd) % nd);
            for (size_t k = 0; k < movable.size(); k++) {
                if (home.tensor_device[movable[k].idx[0]] != d) continue;
                (nxt.alloc.tensor_device[movable[k].idx[0]] == d ? on : off).push_back(k);
            }
            if (on.empty() || off.empty()) continue;
            set_group(nxt.alloc, movable[on[(size_t) (uni(rng) * on.size()) % on.size()]], fit_advisor_allocation::DEV_CPU);
            set_group(nxt.alloc, movable[off[(size_t) (uni(rng) * off.size()) % off.size()]], d);
        } else if (mv < 0.93 && nd > 1) {
            // move one layer across the boundary between two adjacent devices
            const size_t d = (size_t) (uni(rng) * (nd - 1)) % (nd - 1);
            const bool to_right = uni(rng) < 0.5;
            std::vector<uint32_t> part = nxt.alloc.layers_per_device;
            if (to_right ? part[d] == 0 : part[d + 1] <= 1) continue; // keep the output layer with the last device
            if (to_right) { part[d]--; part[d + 1]++; } else { part[d]++; part[d + 1]--; }
            // tensors keep their on/off status relative to their (new) home
            fit_advisor_allocation old_home = fit_advisor_allocation::from_layer_split(inv, device_bufts, nxt.alloc.layers_per_device, opts.n_ctx, nxt.alloc.n_slots, nxt.alloc.layer_home);
            fit_advisor_allocation new_home = fit_advisor_allocation::from_layer_split(inv, device_bufts, part, opts.n_ctx, nxt.alloc.n_slots, nxt.alloc.layer_home);
            for (size_t i = 0; i < inv.tensors.size(); i++) {
                const bool on_home = nxt.alloc.tensor_device[i] == old_home.tensor_device[i];
                nxt.alloc.tensor_device[i] = on_home ? new_home.tensor_device[i] : fit_advisor_allocation::DEV_CPU;
            }
            nxt.alloc.layers_per_device = part;
        } else if (mv < 0.9525 && nd > 1 && nxt.alloc.draft_mtp && inv.n_layer_nextn > 0) {
            // the draft layers and the output head to another device, with the partition boundary of that device
            // shifted so that as many trunk layers leave it as their bytes need: the head's verification rows and the
            // draft decodes on a fast card pay for a layer or two on a slower one, and the move cannot be made one
            // layer at a time because the first half does not fit
            const uint32_t n_layer_all = S.n_layer_all;
            const int h = nxt.alloc.layer_device(inv.n_layer, n_layer_all);
            if (h == fit_advisor_allocation::DEV_CPU) continue;
            const int d = (int) ((h + 1 + (size_t) (uni(rng) * (nd - 1)) % (nd - 1)) % nd);
            std::vector<int64_t> over;
            const bool known = S.memory_over(nxt.alloc, nxt.wl, over);
            size_t moving = 0; // bytes the draft layers and the head bring to d
            for (size_t i = 0; i < inv.tensors.size(); i++) {
                const auto & t = inv.tensors[i];
                const bool of_draft = t.layer >= (int32_t) inv.n_layer || t.kind == FIT_ADVISOR_TENSOR_OUTPUT || t.kind == FIT_ADVISOR_TENSOR_GLOBAL;
                if (of_draft && nxt.alloc.tensor_device[i] != d && nxt.alloc.tensor_device[i] == nxt.alloc.layer_device(t.layer >= 0 ? (uint32_t) t.layer : n_layer_all, n_layer_all)) {
                    moving += t.nbytes;
                }
            }
            for (uint32_t il = inv.n_layer; il <= n_layer_all; il++) {
                nxt.alloc = nxt.alloc.with_layer_home(inv, il, d);
            }
            // shift the boundary of d towards its neighbour until the bytes fit, at most a few layers
            int64_t slack = known && (size_t) d < over.size() ? -over[d] - (int64_t) (64 * UNIT) : 0;
            for (int k = 0; k < 4 && slack < (int64_t) moving; k++) {
                std::vector<uint32_t> part = nxt.alloc.layers_per_device;
                const bool to_right = (size_t) d + 1 < nd;
                const size_t nb = to_right ? (size_t) d + 1 : (size_t) d - 1;
                if (part[d] <= 1) break;
                // the layer at d's edge next to the neighbour leaves: its tensors on d are the bytes freed
                uint32_t first = (uint32_t) n_layer_all + 1 - nxt.alloc.n_gpu_layers();
                for (size_t e = 0; e < (size_t) d; e++) first += part[e];
                const uint32_t il_edge = to_right ? first + part[d] - 1 : first;
                int64_t freed = 0;
                for (size_t i = 0; i < inv.tensors.size(); i++) {
                    if (inv.tensors[i].layer == (int32_t) il_edge && nxt.alloc.tensor_device[i] == d) freed += inv.tensors[i].nbytes;
                }
                part[d]--; part[nb]++;
                fit_advisor_allocation old_home = fit_advisor_allocation::from_layer_split(inv, device_bufts, nxt.alloc.layers_per_device, opts.n_ctx, nxt.alloc.n_slots, nxt.alloc.layer_home);
                fit_advisor_allocation new_home = fit_advisor_allocation::from_layer_split(inv, device_bufts, part, opts.n_ctx, nxt.alloc.n_slots, nxt.alloc.layer_home);
                for (size_t i = 0; i < inv.tensors.size(); i++) {
                    const bool on_home = nxt.alloc.tensor_device[i] == old_home.tensor_device[i];
                    nxt.alloc.tensor_device[i] = on_home ? new_home.tensor_device[i] : fit_advisor_allocation::DEV_CPU;
                }
                nxt.alloc.layers_per_device = part;
                slack += freed;
            }
            n_exchange_tried++;
        } else if (mv < 0.955 && nd > 1) {
            // re-home one whole layer on another device: its KV cache, state and pinned ops go with it (-old).
            // MTP layers and the output layer sit at the end of the numbering, so they get half the draws
            const uint32_t n_layer_all = S.n_layer_all;
            uint32_t il;
            if (uni(rng) < 0.5) {
                il = inv.n_layer + (uint32_t) (uni(rng) * (inv.n_layer_nextn + 1)) % (inv.n_layer_nextn + 1); // an MTP layer or the output
            } else {
                il = (uint32_t) (uni(rng) * (n_layer_all + 1)) % (n_layer_all + 1);
            }
            const int h = nxt.alloc.layer_device(il, n_layer_all);
            if (h == fit_advisor_allocation::DEV_CPU) continue; // the leading CPU block is the partition's business
            const int d = (int) ((h + 1 + (size_t) (uni(rng) * (nd - 1)) % (nd - 1)) % nd);
            nxt.alloc = nxt.alloc.with_layer_home(inv, il, d);
            n_rehome_tried++;
        } else if (mv < 0.96 && opts.offload_options.size() > 1) {
            // step one device's op offload threshold: from which batch it takes CPU-resident weights of its layers;
            // a device that declines leaves them to the first willing device and keeps a small compute buffer
            const auto & oo = opts.offload_options;
            const size_t d = (size_t) (uni(rng) * nd) % nd;
            if (nxt.alloc.op_offload_min_batch_dev.size() < nd) nxt.alloc.op_offload_min_batch_dev.resize(nd, 0);
            size_t k = std::find(oo.begin(), oo.end(), nxt.alloc.op_offload_min_batch_dev[d]) - oo.begin();
            if (k >= oo.size()) continue;
            k = uni(rng) < 0.5 ? (k == 0 ? 1 : k - 1) : (k + 1 >= oo.size() ? k - 1 : k + 1);
            if (k >= oo.size()) continue;
            nxt.alloc.op_offload_min_batch_dev[d] = oo[k];
            nxt.wl = S.workload(nxt.alloc);
            n_offload_tried++;
        } else if (mv < 0.965 && S.mtp_searchable()) {
            // drafting on or off: with it the MTP layers cost memory and a verification batch replaces single tokens
            nxt.alloc.draft_mtp = !nxt.alloc.draft_mtp;
            nxt.wl = S.workload(nxt.alloc);
            n_mtp_tried++;
        } else if (mv < 0.9675 && opts.search_flash_attn) {
            // flash attention on or off: the attention path and the compute buffer size change
            nxt.alloc.flash_attn = nxt.alloc.flash_attn == 1 ? 0 : 1;
            n_attn_tried++;
        } else if (mv < 0.97 && opts.search_kv_offload) {
            // the whole KV cache on the host with attention on the CPU, freeing the cards for weights
            nxt.alloc.no_kv_offload = !nxt.alloc.no_kv_offload;
            n_attn_tried++;
        } else if (mv < 0.97) {
            // step the ubatch
            const auto & ubs = opts.ubatch_options;
            size_t k = std::find(ubs.begin(), ubs.end(), nxt.alloc.n_ubatch) - ubs.begin();
            if (k >= ubs.size()) continue;
            k = uni(rng) < 0.5 ? (k == 0 ? 1 : k - 1) : (k + 1 >= ubs.size() ? k - 1 : k + 1);
            if (k >= ubs.size()) continue;
            nxt.alloc.n_ubatch = ubs[k];
            nxt.wl = S.workload(nxt.alloc);
        } else {
            // step the slot count
            uint32_t s = nxt.alloc.n_slots;
            s = uni(rng) < 0.5 ? std::max<uint32_t>(1, s / 2) : std::min<uint32_t>(std::max<uint32_t>(1, opts.max_slots), s * 2);
            if (s == nxt.alloc.n_slots) continue;
            nxt.alloc.n_slots = s;
        }

        // a key never probed needs one probe for its overheads
        const std::string key = alloc_key(nxt.alloc);
        if (!proj_by_key.count(key)) {
            const fit_advisor_projection & pj = S.probe_alloc(nxt.alloc, alloc_name(nxt.alloc));
            if (!pj.ok) continue;
            proj_by_key[key] = pj;
        }
        if (!S.evaluate(nxt, proj_by_key[key])) continue;

        const double delta = nxt.objective - cur.objective;
        if (mv < 0.15 && trace_tensor) {
            for (size_t i = 0; i < inv.tensors.size(); i++) {
                if (nxt.alloc.tensor_device[i] != cur.alloc.tensor_device[i] && inv.tensors[i].name.find(trace_tensor) != std::string::npos) {
                    const fit_advisor_cost c0 = fit_advisor_cost_estimate(inv, cur.alloc, proj_by_key[key], gp, cost_devs, pairs, cur.wl);
                    const fit_advisor_cost c1 = fit_advisor_cost_estimate(inv, nxt.alloc, proj_by_key[key], gp, cost_devs, pairs, nxt.wl);
                    LOG_INF("%s: iter %d T %.4f s: move %s %s -> %s: delta %+.4f s (weights %+.1f attn %+.1f overhead %+.1f boundary %+.1f us/gen step; pp %.0f -> %.0f tok/s)\n", __func__,
                        it, T * 1e-6, inv.tensors[i].name.c_str(),
                        cur.alloc.tensor_device[i] < 0 ? "CPU" : device_bufts[cur.alloc.tensor_device[i]].c_str(),
                        nxt.alloc.tensor_device[i] < 0 ? "CPU" : device_bufts[nxt.alloc.tensor_device[i]].c_str(),
                        delta * 1e-6, c1.step_weights_us - c0.step_weights_us, c1.step_attn_us - c0.step_attn_us,
                        c1.step_overhead_us - c0.step_overhead_us, c1.step_boundary_us - c0.step_boundary_us,
                        c0.prompt_tokens_per_s, c1.prompt_tokens_per_s);
                }
            }
        }
        if (mv < 0.15 && n_single_tried <= 12) {
            // trace the first single-tensor moves: which tensor, where, and what the model thinks of it
            for (size_t i = 0; i < inv.tensors.size(); i++) {
                if (nxt.alloc.tensor_device[i] != cur.alloc.tensor_device[i]) {
                    LOG_DBG("%s: move %-34s %s -> %s: cost %.4f -> %.4f s, penalty %.4f -> %.4f s, delta %+.4f s\n", __func__,
                        inv.tensors[i].name.c_str(),
                        cur.alloc.tensor_device[i] < 0 ? "CPU" : device_bufts[cur.alloc.tensor_device[i]].c_str(),
                        nxt.alloc.tensor_device[i] < 0 ? "CPU" : device_bufts[nxt.alloc.tensor_device[i]].c_str(),
                        cur.cost_score * 1e-6, nxt.cost_score * 1e-6, cur.penalty * 1e-6, nxt.penalty * 1e-6, delta * 1e-6);
                }
            }
        }
        if (delta < 0 || (delta > 0 && uni(rng) < std::exp(-delta / T))) {
            cur = nxt;
            accepted++;
            // track the best model-feasible state continuously; it is verified by the loader below, and again at the end
            if (cur.penalty == 0 && cur.objective < best_seen.objective) {
                best_seen = cur;
                best_seen_dirty = true;
                n_improvements++;
            }
        }
        // verify the best state with the real loader at most every 200 iterations, or when the walk is nearly done
        if (best_seen_dirty && (it - last_probe_iter >= 500 || it == iters - 1)) {
            const std::string bkey = alloc_key(best_seen.alloc);
            const fit_advisor_projection & pj = S.probe_alloc(best_seen.alloc, alloc_name(best_seen.alloc));
            last_probe_iter = it;
            best_seen_dirty = false;
            proj_by_key[bkey] = pj;
            searcher::state checked = best_seen;
            const bool ok = pj.ok && pj.fits_all() && S.evaluate(checked, pj) && checked.penalty == 0 && checked.objective < incumbent.objective;
            LOG_DBG("%s: iter %d: verifying best state (model %.3f s vs incumbent %.3f s): %s\n", __func__, it,
                best_seen.objective * 1e-6, incumbent.objective * 1e-6,
                !pj.ok ? "probe failed" : !pj.fits_all() ? "does not fit on probe" : !ok ? "not better after refresh" : "accepted as incumbent");
            if (ok) {
                incumbent = checked;
                incumbent_proj = pj;
            } else {
                // the refreshed overheads say it does not fit: forget it. the walk continues from its current state if
                // that still fits under the refreshed model, else from the incumbent rather than from penalty
                best_seen = incumbent;
                if (cur.penalty == 0) {
                    S.evaluate(cur, proj_by_key.count(key) ? proj_by_key[key] : pj);
                }
                if (cur.penalty > 0) {
                    cur = incumbent;
                }
            }
        }
    }

    LOG_INF("%s: annealing: %d accepted of %d, %d single-tensor moves, %d layer re-homes, %d draft-layer exchanges, %d drafting toggles, %d offload threshold steps and %d attention toggles proposed, %d model improvements over the seed\n", __func__,
        accepted, iters, n_single_tried, n_rehome_tried, n_exchange_tried, n_mtp_tried, n_offload_tried, n_attn_tried, n_improvements);
    GGML_UNUSED(n_single_accepted);

    // fill: from the incumbent, everything CPU-resident that the full model says pays for itself, on any device. the
    // packed state is probed like the one before annealing, refilled with the probed overheads if it does not fit,
    // and the verified incumbent stays if no packed state passes
    {
        const searcher::state before = incumbent;
        for (int round = 0; round < 3; round++) {
            searcher::state st = before;
            const std::string k = alloc_key(st.alloc);
            if (!S.evaluate(st, proj_by_key.count(k) ? proj_by_key[k] : incumbent_proj) || st.penalty > 0) {
                break;
            }
            if (fill_pass(st, proj_by_key, "after annealing") == 0) {
                break;
            }
            const fit_advisor_projection & pj = S.probe_alloc(st.alloc, alloc_name(st.alloc));
            proj_by_key[k] = pj;
            searcher::state checked = st;
            if (pj.ok && pj.fits_all() && S.evaluate(checked, pj) && checked.penalty == 0 && checked.objective < incumbent.objective) {
                incumbent      = checked;
                incumbent_proj = pj;
                break;
            }
            LOG_INF("%s: the filled incumbent does not fit on probe (round %d), refilling with the probed overheads\n", __func__, round + 1);
        }
    }

    // final verification of the incumbent: every incumbent was probed when it was accepted, so this only guards against
    // a probe that reads differently now; the seed is never returned over a verified incumbent
    {
        const fit_advisor_projection & pj = S.probe_alloc(incumbent.alloc, alloc_name(incumbent.alloc));
        if (pj.ok && pj.fits_all()) {
            incumbent_proj = pj;
        } else {
            LOG_WRN("%s: the incumbent does not fit on its final re-probe; it was accepted on an earlier probe and is kept\n", __func__);
        }
    }

    finish_result(best, incumbent.alloc, incumbent.wl, incumbent_proj, inv, gp, cost_devs, pairs, device_bufts, wl_base, S.n_probes, accepted);
    return best;
}
