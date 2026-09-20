#include "search.h"

#include "ggml.h"

#include "log.h"

#include <algorithm>
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

    // request-time gain of placing tensor i on device d instead of the CPU
    double gain(size_t i, int d, const fit_advisor_workload & wl, uint32_t slots) const {
        const auto & t = inv.tensors[i];
        return fit_advisor_tensor_request_us(inv, t, fit_advisor_allocation::DEV_CPU, cost_devs, wl, slots)
             - fit_advisor_tensor_request_us(inv, t, d, cost_devs, wl, slots);
    }

    // movable groups: the expert tensors of a layer move together (a partially moved layer still costs its split),
    // every other tensor is its own group. only groups with a positive, measured gain are candidates to leave their
    // device; the rest (norms, biases, unmeasured types) always stay with their layer
    struct group {
        std::vector<size_t> idx;
        size_t bytes = 0;
    };

    // a tensor the search may consider moving: large enough to matter and of a type with a measured rate on every
    // device, so its cost on either side is known. norms, biases, recurrent-state parameters and conv weights fail
    // this and always stay with their layer: moving them buys nothing and drags their ops across the bus
    bool tensor_movable(size_t i) const {
        const auto & t = inv.tensors[i];
        if (t.nbytes < (size_t) UNIT) {
            return false;
        }
        for (const auto & d : cost_devs) {
            if (!d.meas) {
                return false;
            }
            const auto it = d.meas->matmul.find(ggml_type_name(t.type));
            if (it == d.meas->matmul.end() || !it->second.supported || it->second.bytes_per_s <= 0) {
                return false;
            }
        }
        return true;
    }

    std::vector<group> build_groups(const fit_advisor_workload & wl) const {
        std::map<int32_t, group> exps_by_layer;
        std::vector<group> ret;
        for (size_t i = 0; i < inv.tensors.size(); i++) {
            const auto & t = inv.tensors[i];
            if (!tensor_counts(i, wl) || t.layer < 0 || !tensor_movable(i)) {
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
                s.penalty += over[d] / (double) UNIT * 2000.0; // 2 ms of request time per MiB over budget
            }
        }
        s.objective = s.cost_score + s.penalty;
        return true;
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

    // movable groups: those with a positive gain on their home
    std::vector<searcher::group> movable;
    {
        fit_advisor_allocation home = fit_advisor_allocation::from_layer_split(inv, device_bufts, cur.alloc.layers_per_device, opts.n_ctx, cur.alloc.n_slots);
        for (const auto & g : S.build_groups(cur.wl)) {
            const int h = home.tensor_device[g.idx[0]];
            if (h >= 0 && S.group_gain(g, h, cur.wl, cur.alloc.n_slots) > 0) {
                movable.push_back(g);
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

    const double T0 = std::max(1.0, 0.02 * std::fabs(cur.cost_score));
    const double T1 = std::max(0.01, 0.0001 * std::fabs(cur.cost_score));
    const int iters = std::max(0, opts.anneal_iters);
    int accepted = 0;
    int last_probe_iter = 0;

    for (int it = 0; it < iters && !movable.empty(); it++) {
        const double T = T0 * std::pow(T1 / T0, (double) it / std::max(1, iters));
        searcher::state nxt = cur;
        const double mv = uni(rng);

        if (mv < 0.10) {
            // toggle a single tensor of a multi-tensor group: a partial layer, worth it only if the split is cheap
            const searcher::group & g = movable[(size_t) (uni(rng) * movable.size()) % movable.size()];
            if (g.idx.size() < 2) continue;
            const size_t i = g.idx[(size_t) (uni(rng) * g.idx.size()) % g.idx.size()];
            fit_advisor_allocation home = fit_advisor_allocation::from_layer_split(inv, device_bufts, nxt.alloc.layers_per_device, opts.n_ctx, nxt.alloc.n_slots);
            nxt.alloc.tensor_device[i] = nxt.alloc.tensor_device[i] == fit_advisor_allocation::DEV_CPU ? home.tensor_device[i] : fit_advisor_allocation::DEV_CPU;
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
        if (delta < 0 || (delta > 0 && uni(rng) < std::exp(-delta / T))) {
            cur = nxt;
            accepted++;
            if (cur.penalty == 0 && cur.objective < incumbent.objective) {
                // verify improved incumbents with the real loader, at most every 200 iterations
                if (it - last_probe_iter >= 200 || it == iters - 1) {
                    const fit_advisor_projection & pj = S.probe_alloc(cur.alloc, cell_name(cur.alloc.layers_per_device, cur.alloc.n_ubatch, cur.alloc.n_slots));
                    last_probe_iter = it;
                    proj_by_key[key] = pj;
                    if (!pj.ok || !pj.fits_all()) {
                        S.evaluate(cur, pj); // the refreshed overheads penalise it now
                        continue;
                    }
                    S.evaluate(cur, pj);
                    if (cur.objective < incumbent.objective) {
                        incumbent = cur;
                        incumbent_proj = pj;
                    }
                }
            }
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
