#include "search.h"

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

std::string cell_name(const std::vector<uint32_t> & part, uint32_t ub, uint32_t slots) {
    std::string n = "search";
    for (size_t d = 0; d < part.size(); d++) {
        n += (d ? "/" : "-") + std::to_string(part[d]);
    }
    return n + "-ub" + std::to_string(ub) + "-np" + std::to_string(slots);
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

    fit_advisor_workload workload(uint32_t ub) const {
        fit_advisor_workload wl = wl_base;
        wl.n_ubatch = ub;
        return wl;
    }

    bool tensor_counts(size_t i, const fit_advisor_workload & wl) const {
        return !(inv.tensors[i].layer >= (int32_t) inv.n_layer && !wl.use_mtp);
    }

    // probe an allocation, record its overheads for its key, count it
    const fit_advisor_projection & probe_alloc(const fit_advisor_allocation & a, const std::string & name) {
        const fit_advisor_projection & pj = probe.run(a.to_candidate(inv, device_bufts, name));
        n_probes++;
        if (pj.ok) {
            memory_model::entry e;
            e.ok = true;
            for (size_t d = 0; d < nd && d < pj.devices.size(); d++) {
                e.overhead.push_back((int64_t) (pj.devices[d].context + pj.devices[d].compute + pj.devices[d].scratch));
                e.free.push_back(pj.devices[d].free);
                e.margin.push_back(pj.devices[d].margin);
            }
            mem.by_key[cell_key(a.layers_per_device, a.n_ubatch, a.n_slots)] = e;
        }
        return pj;
    }

    // linear memory model: weights on each device plus the probed overhead for the key; returns the over-budget bytes
    // per device (negative = room), or false if the key was never probed
    bool memory_over(const fit_advisor_allocation & a, const fit_advisor_workload & wl, std::vector<int64_t> & over) const {
        const auto it = mem.by_key.find(cell_key(a.layers_per_device, a.n_ubatch, a.n_slots));
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
        return fit_advisor_tensor_request_us(inv, gp, i, fit_advisor_allocation::DEV_CPU, cost_devs, wl, slots)
             - fit_advisor_tensor_request_us(inv, gp, i, d, cost_devs, wl, slots);
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
        const fit_advisor_workload wl = workload(ub);
        const std::string name = cell_name(part, ub, slots);

        fit_advisor_allocation base = fit_advisor_allocation::from_layer_split(inv, device_bufts, part, opts.n_ctx, slots);
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
            for (size_t d = 0; d < nd; d++) {
                per[d] = (uint32_t) std::floor(S.ngl_max * w[d] / sum);
                assigned += per[d];
            }
            per[nd - 1] += S.ngl_max - assigned;
            return per;
        };
        partitions.push_back(split(w_free));
        const std::vector<uint32_t> even = split(std::vector<double>(nd, 1.0));
        if (even != partitions[0]) {
            partitions.push_back(even);
        }
    }
    std::vector<uint32_t> slot_options;
    for (uint32_t s = 1; s <= std::max<uint32_t>(1, opts.max_slots); s *= 2) {
        slot_options.push_back(s);
    }

    searcher::cell best_cell;
    std::vector<uint32_t> best_part;
    uint32_t best_ub = 0;
    for (const auto & part : partitions) {
        for (const uint32_t ub : opts.ubatch_options) {
            for (const uint32_t slots : slot_options) {
                const std::string name = cell_name(part, ub, slots);
                searcher::cell r = S.solve_cell(part, ub, slots);
                best.n_cells++;
                if (!r.fits || !r.cost.ok) {
                    LOG_INF("%s: seed %-28s does not fit\n", __func__, name.c_str());
                    continue;
                }
                const fit_advisor_workload wl = S.workload(ub);
                const bool better = !best_cell.fits || S.score(r.cost, wl) < S.score(best_cell.cost, S.workload(best_ub));
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
    cur.wl    = S.workload(best_ub);
    std::map<std::string, fit_advisor_projection> proj_by_key; // last probe per key, for the cost model's KV figures
    proj_by_key[cell_key(cur.alloc.layers_per_device, cur.alloc.n_ubatch, cur.alloc.n_slots)] = best_cell.proj;
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
        const std::string key0 = cell_key(cur.alloc.layers_per_device, cur.alloc.n_ubatch, cur.alloc.n_slots);
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
        fit_advisor_allocation home = fit_advisor_allocation::from_layer_split(inv, device_bufts, cur.alloc.layers_per_device, opts.n_ctx, cur.alloc.n_slots);
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
    auto set_group = [&](fit_advisor_allocation & a, const searcher::group & g, int dev) {
        for (const size_t i : g.idx) {
            a.tensor_device[i] = dev;
        }
    };

    int n_improvements = 0;
    int n_single_tried = 0, n_single_accepted = 0;
    const char * trace_tensor = getenv("FIT_ADVISOR_TRACE"); // substring of a tensor name whose moves are logged
    const double T0 = std::max(1.0, 0.005 * std::fabs(cur.cost_score));
    const double T1 = std::max(0.01, 0.00002 * std::fabs(cur.cost_score));
    const int iters = std::max(0, opts.anneal_iters);
    int accepted = 0;
    int last_probe_iter = 0;

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
            fit_advisor_allocation home = fit_advisor_allocation::from_layer_split(inv, device_bufts, nxt.alloc.layers_per_device, opts.n_ctx, nxt.alloc.n_slots);
            nxt.alloc.tensor_device[i] = nxt.alloc.tensor_device[i] == fit_advisor_allocation::DEV_CPU ? home.tensor_device[i] : fit_advisor_allocation::DEV_CPU;
            n_single_tried++;
        } else if (mv < 0.55) {
            // toggle one group between its home and the CPU
            const searcher::group & g = movable[(size_t) (uni(rng) * movable.size()) % movable.size()];
            fit_advisor_allocation home = fit_advisor_allocation::from_layer_split(inv, device_bufts, nxt.alloc.layers_per_device, opts.n_ctx, nxt.alloc.n_slots);
            const int h = home.tensor_device[g.idx[0]];
            set_group(nxt.alloc, g, nxt.alloc.tensor_device[g.idx[0]] == fit_advisor_allocation::DEV_CPU ? h : fit_advisor_allocation::DEV_CPU);
        } else if (mv < 0.85) {
            // swap one on-device group with one CPU group of the same home
            std::vector<size_t> on, off;
            fit_advisor_allocation home = fit_advisor_allocation::from_layer_split(inv, device_bufts, nxt.alloc.layers_per_device, opts.n_ctx, nxt.alloc.n_slots);
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
            fit_advisor_allocation old_home = fit_advisor_allocation::from_layer_split(inv, device_bufts, nxt.alloc.layers_per_device, opts.n_ctx, nxt.alloc.n_slots);
            fit_advisor_allocation new_home = fit_advisor_allocation::from_layer_split(inv, device_bufts, part, opts.n_ctx, nxt.alloc.n_slots);
            for (size_t i = 0; i < inv.tensors.size(); i++) {
                const bool on_home = nxt.alloc.tensor_device[i] == old_home.tensor_device[i];
                nxt.alloc.tensor_device[i] = on_home ? new_home.tensor_device[i] : fit_advisor_allocation::DEV_CPU;
            }
            nxt.alloc.layers_per_device = part;
        } else if (mv < 0.97) {
            // step the ubatch
            const auto & ubs = opts.ubatch_options;
            size_t k = std::find(ubs.begin(), ubs.end(), nxt.alloc.n_ubatch) - ubs.begin();
            if (k >= ubs.size()) continue;
            k = uni(rng) < 0.5 ? (k == 0 ? 1 : k - 1) : (k + 1 >= ubs.size() ? k - 1 : k + 1);
            if (k >= ubs.size()) continue;
            nxt.alloc.n_ubatch = ubs[k];
            nxt.wl = S.workload(ubs[k]);
        } else {
            // step the slot count
            uint32_t s = nxt.alloc.n_slots;
            s = uni(rng) < 0.5 ? std::max<uint32_t>(1, s / 2) : std::min<uint32_t>(std::max<uint32_t>(1, opts.max_slots), s * 2);
            if (s == nxt.alloc.n_slots) continue;
            nxt.alloc.n_slots = s;
        }

        // a key never probed needs one probe for its overheads
        const std::string key = cell_key(nxt.alloc.layers_per_device, nxt.alloc.n_ubatch, nxt.alloc.n_slots);
        if (!proj_by_key.count(key)) {
            const fit_advisor_projection & pj = S.probe_alloc(nxt.alloc, cell_name(nxt.alloc.layers_per_device, nxt.alloc.n_ubatch, nxt.alloc.n_slots));
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
        if (best_seen_dirty && (it - last_probe_iter >= 200 || it == iters - 1)) {
            const std::string bkey = cell_key(best_seen.alloc.layers_per_device, best_seen.alloc.n_ubatch, best_seen.alloc.n_slots);
            const fit_advisor_projection & pj = S.probe_alloc(best_seen.alloc, cell_name(best_seen.alloc.layers_per_device, best_seen.alloc.n_ubatch, best_seen.alloc.n_slots));
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
                // the refreshed overheads say it does not fit: forget it and let the walk continue from the incumbent
                best_seen = incumbent;
                if (cur.penalty == 0) {
                    S.evaluate(cur, proj_by_key.count(key) ? proj_by_key[key] : pj);
                }
            }
        }
    }

    LOG_INF("%s: annealing: %d accepted of %d, %d single-tensor moves proposed, %d model improvements over the seed\n", __func__,
        accepted, iters, n_single_tried, n_improvements);
    GGML_UNUSED(n_single_accepted);

    // fill: from the incumbent, add every CPU-resident tensor that the model says pays for itself, best gain per byte
    // first, while the memory model has room; the probe verifies the result below
    {
        searcher::state st = incumbent;
        fit_advisor_allocation home = fit_advisor_allocation::from_layer_split(inv, device_bufts, st.alloc.layers_per_device, opts.n_ctx, st.alloc.n_slots);
        struct cand { size_t i; double density; };
        std::vector<cand> cands;
        // every CPU-resident single is tried, whatever its weight-only gain: the full model with the excursion cost
        // decides, so a tensor the walk parked on the CPU at high temperature comes back if that was a bad trade
        for (const size_t i : singles) {
            if (st.alloc.tensor_device[i] != fit_advisor_allocation::DEV_CPU || home.tensor_device[i] < 0) continue;
            const double g = S.gain(i, home.tensor_device[i], st.wl, st.alloc.n_slots);
            cands.push_back({ i, g / (double) inv.tensors[i].nbytes });
        }
        std::sort(cands.begin(), cands.end(), [](const cand & a, const cand & b) { return a.density > b.density; });
        int n_filled = 0;
        const std::string fkey = cell_key(st.alloc.layers_per_device, st.alloc.n_ubatch, st.alloc.n_slots);
        for (const cand & c : cands) {
            searcher::state nxt = st;
            nxt.alloc.tensor_device[c.i] = home.tensor_device[c.i];
            const bool feasible = S.evaluate(nxt, proj_by_key[fkey]);
            if (trace_tensor && inv.tensors[c.i].name.find(trace_tensor) != std::string::npos) {
                LOG_INF("%s: fill %-34s CPU -> %s: %s, cost %.4f -> %.4f s, weight-only gain %+.4f s\n", __func__,
                    inv.tensors[c.i].name.c_str(), device_bufts[home.tensor_device[c.i]].c_str(),
                    feasible ? "fits" : "over budget", st.cost_score * 1e-6, nxt.cost_score * 1e-6,
                    c.density * (double) inv.tensors[c.i].nbytes * 1e-6);
            }
            if (feasible && nxt.objective < st.objective) {
                st = nxt;
                n_filled++;
            }
        }
        if (n_filled > 0) {
            LOG_INF("%s: fill pass placed %d more tensors, model %.3f s -> %.3f s\n", __func__, n_filled, incumbent.objective * 1e-6, st.objective * 1e-6);
            incumbent = st;
        }
    }

    // final verification of the incumbent
    {
        const fit_advisor_projection & pj = S.probe_alloc(incumbent.alloc, cell_name(incumbent.alloc.layers_per_device, incumbent.alloc.n_ubatch, incumbent.alloc.n_slots));
        if (pj.ok && pj.fits_all()) {
            incumbent_proj = pj;
        } else {
            LOG_WRN("%s: annealed incumbent does not fit on re-probe, falling back to the seed\n", __func__);
            incumbent.alloc = best_cell.alloc;
            incumbent.wl    = S.workload(best_ub);
            incumbent_proj  = best_cell.proj;
        }
    }

    best.ok    = true;
    best.name  = cell_name(incumbent.alloc.layers_per_device, incumbent.alloc.n_ubatch, incumbent.alloc.n_slots);
    best.alloc = incumbent.alloc;
    best.wl    = incumbent.wl;
    best.proj  = incumbent_proj;
    best.cost  = fit_advisor_cost_estimate(inv, best.alloc, best.proj, gp, cost_devs, pairs, best.wl);
    best.cand  = best.alloc.to_candidate(inv, device_bufts, best.name);
    best.n_probes = S.n_probes;
    best.n_anneal_accepted = accepted;
    return best;
}
