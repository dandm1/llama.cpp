#include "solve.h"

#include "log.h"

#include <algorithm>
#include <cmath>
#include <map>

#ifdef FIT_ADVISOR_HIGHS
#include "Highs.h"
#endif

bool fit_advisor_solver_available() {
#ifdef FIT_ADVISOR_HIGHS
    return true;
#else
    return false;
#endif
}

#ifndef FIT_ADVISOR_HIGHS
fit_advisor_solve_result fit_advisor_solve_placement(const fit_advisor_solve_input & in) {
    GGML_UNUSED(in);
    fit_advisor_solve_result r;
    r.error = "built without HiGHS";
    return r;
}
#else

namespace {

constexpr int64_t UNIT = 1024 * 1024;

struct group {
    std::vector<size_t> idx;
    size_t bytes = 0;
    int    home  = 0; // allocation device of the group's layer
};

} // namespace

fit_advisor_solve_result fit_advisor_solve_placement(const fit_advisor_solve_input & in) {
    fit_advisor_solve_result r;
    if (!in.inv || !in.gp || !in.devices || !in.pairs || !in.wl || !in.device_bufts) {
        r.error = "missing inputs";
        return r;
    }
    const fit_advisor_inventory & inv = *in.inv;
    const fit_advisor_graph_profile & gp = *in.gp;
    const std::vector<fit_advisor_cost_device> & devices = *in.devices;
    const fit_advisor_workload & wl = *in.wl;
    const size_t nd = in.device_bufts->size();            // allocation devices (cards)
    const size_t nk = nd + 1;                             // placement choices: the cards, then the CPU
    const uint32_t n_layer_all = inv.n_layer + inv.n_layer_nextn;
    const uint32_t n_slots     = std::max<uint32_t>(1, in.base.n_slots);
    if (in.capacity.size() < nd || in.base.tensor_device.size() != inv.tensors.size()) {
        r.error = "capacity or base allocation of the wrong size";
        return r;
    }
    auto dev_of = [&](size_t k) -> int { return k < nd ? (int) k : fit_advisor_allocation::DEV_CPU; };

    const int64_t t0 = ggml_time_us();

    // ---- what moves: the expert stacks of a layer as one group, every other large measured tensor on its own;
    // the draft block (MTP layers and the output head) as a unit with its own home; everything else stays put
    auto counts = [&](size_t i) {
        return !(inv.tensors[i].layer >= (int32_t) inv.n_layer && !wl.use_mtp);
    };
    auto used = [&](size_t i) {
        return i < gp.use_tg.size() && gp.use_tg[i].op != 0;
    };
    auto decidable = [&](size_t i, int home) {
        if (i >= gp.use_tg.size() || i >= gp.use_pp.size()) {
            return false;
        }
        const auto & t = inv.tensors[i];
        return fit_advisor_tensor_cost_known(inv, t, gp.use_tg[i], home, devices)
            && fit_advisor_tensor_cost_known(inv, t, gp.use_tg[i], fit_advisor_allocation::DEV_CPU, devices)
            && fit_advisor_tensor_cost_known(inv, t, gp.use_pp[i], home, devices)
            && fit_advisor_tensor_cost_known(inv, t, gp.use_pp[i], fit_advisor_allocation::DEV_CPU, devices);
    };
    auto of_draft_block = [&](size_t i) {
        const auto & t = inv.tensors[i];
        return t.layer >= (int32_t) inv.n_layer || t.kind == FIT_ADVISOR_TENSOR_OUTPUT || t.kind == FIT_ADVISOR_TENSOR_GLOBAL;
    };

    std::vector<group> groups;
    std::map<int32_t, group> exps_by_layer;
    std::vector<int64_t> fixed_bytes(nd, 0); // bytes of tensors that stay where the base puts them, per card
    std::vector<size_t>  block_idx;          // the draft block's tensors
    int64_t block_bytes = 0;
    const bool move_block = in.move_draft_block && wl.use_mtp && inv.n_layer_nextn > 0;
    for (size_t i = 0; i < inv.tensors.size(); i++) {
        const auto & t = inv.tensors[i];
        if (!counts(i)) {
            continue; // not loaded
        }
        if (move_block && of_draft_block(i)) {
            block_idx.push_back(i);
            block_bytes += (int64_t) t.nbytes;
            continue;
        }
        const int home = t.layer >= 0 ? in.base.layer_device((uint32_t) t.layer, n_layer_all) : in.base.layer_device(n_layer_all, n_layer_all);
        const bool movable = t.layer >= 0 && used(i) && t.nbytes >= (size_t) UNIT && home >= 0 && decidable(i, home);
        if (!movable) {
            const int d = in.base.tensor_device[i];
            if (d >= 0 && (size_t) d < nd) {
                fixed_bytes[d] += (int64_t) t.nbytes;
            }
            continue;
        }
        if (t.kind == FIT_ADVISOR_TENSOR_FFN_EXPS) {
            auto & g = exps_by_layer[t.layer];
            g.idx.push_back(i);
            g.bytes += t.nbytes;
            g.home = home;
        } else {
            groups.push_back({ { i }, t.nbytes, home });
        }
    }
    for (auto & [il, g] : exps_by_layer) {
        groups.push_back(g);
    }
    r.n_groups = (int) groups.size();

    // ---- costs: request time of a group on each choice, plus the excursion penalty when it runs away from home
    const uint32_t batch_gen = std::max<uint32_t>(1, std::min(wl.concurrency, n_slots));
    const uint32_t n_ub      = std::max<uint32_t>(1, wl.n_ubatch);
    const double   n_pp      = std::ceil((double) wl.prompt_tokens / n_ub);
    const uint32_t depth     = wl.use_mtp && inv.n_layer_nextn > 0 ? wl.mtp_draft_n_default() : 0;
    const double   n_gen     = wl.gen_tokens / wl.tokens_per_step(depth);
    const uint32_t batch_ver = batch_gen * (1 + depth);
    auto excursion_us = [&](int home, int dev) -> double {
        // the op's activations go to the device holding the weight and back, each step; at the prompt batch a CPU
        // weight is copied to the home instead (priced in the tensor's own cost), a weight on another card is not
        if (dev == home) {
            return 0;
        }
        const double gen = fit_advisor_excursion_us(inv, devices, *in.pairs, home, dev, batch_ver) + wl.split_extra_us;
        double pp = 0;
        if (dev >= 0) {
            pp = fit_advisor_excursion_us(inv, devices, *in.pairs, home, dev, n_ub) + wl.split_extra_us;
        }
        return n_gen * gen + n_pp * pp;
    };
    std::vector<double> cost(groups.size() * nk, 0.0);
    for (size_t g = 0; g < groups.size(); g++) {
        for (size_t k = 0; k < nk; k++) {
            const int dev = dev_of(k);
            double c = 0;
            for (const size_t i : groups[g].idx) {
                c += fit_advisor_tensor_request_us(inv, gp, i, dev, devices, wl, n_slots, groups[g].home);
            }
            cost[g * nk + k] = c + excursion_us(groups[g].home, dev);
        }
    }
    // the draft block on each card: its tensors' request time there, the draft decode's overhead on that device, and
    // the crossing from the last trunk layer's device when it differs
    std::vector<double> block_cost(nd, 0.0);
    const int last_trunk_dev = in.base.layer_device(inv.n_layer > 0 ? inv.n_layer - 1 : 0, n_layer_all);
    if (move_block) {
        for (size_t k = 0; k < nd; k++) {
            double c = 0;
            for (const size_t i : block_idx) {
                c += fit_advisor_tensor_request_us(inv, gp, i, (int) k, devices, wl, n_slots, (int) k);
            }
            c += n_gen * depth * wl.mtp_draft_extra_for((int) k, nd);
            if ((int) k != last_trunk_dev) {
                c += n_gen * (fit_advisor_hop_us(inv, devices, *in.pairs, last_trunk_dev, (int) k, batch_ver) + wl.split_extra_us)
                   + n_pp  * (fit_advisor_hop_us(inv, devices, *in.pairs, last_trunk_dev, (int) k, n_ub) + wl.split_extra_us);
            }
            block_cost[k] = c;
        }
    }

    // ---- the program
    const int n_x = (int) (groups.size() * nk);
    const int n_h = move_block ? (int) nd : 0;
    const int n_cols = n_x + n_h;
    r.n_vars = n_cols;
    HighsModel model;
    HighsLp & lp = model.lp_;
    lp.num_col_ = n_cols;
    lp.col_cost_.assign(n_cols, 0.0);
    lp.col_lower_.assign(n_cols, 0.0);
    lp.col_upper_.assign(n_cols, 1.0);
    lp.integrality_.assign(n_cols, HighsVarType::kInteger);
    for (int c = 0; c < n_x; c++) {
        lp.col_cost_[c] = cost[c];
    }
    for (int k = 0; k < n_h; k++) {
        lp.col_cost_[n_x + k] = block_cost[k];
    }
    // rows: one choice per group, one choice for the block, the capacity of every card. built row-wise
    std::vector<double>  row_lower, row_upper;
    std::vector<HighsInt> a_start, a_index;
    std::vector<double>  a_value;
    auto begin_row = [&](double lo, double hi) { row_lower.push_back(lo); row_upper.push_back(hi); a_start.push_back((HighsInt) a_index.size()); };
    for (size_t g = 0; g < groups.size(); g++) {
        begin_row(1.0, 1.0);
        for (size_t k = 0; k < nk; k++) {
            a_index.push_back((HighsInt) (g * nk + k));
            a_value.push_back(1.0);
        }
    }
    if (move_block) {
        begin_row(1.0, 1.0);
        for (int k = 0; k < n_h; k++) {
            a_index.push_back((HighsInt) (n_x + k));
            a_value.push_back(1.0);
        }
    }
    for (size_t d = 0; d < nd; d++) {
        const double cap = (double) (in.capacity[d] - fixed_bytes[d]) / UNIT;
        begin_row(-kHighsInf, cap);
        for (size_t g = 0; g < groups.size(); g++) {
            a_index.push_back((HighsInt) (g * nk + d));
            a_value.push_back((double) groups[g].bytes / UNIT);
        }
        if (move_block) {
            a_index.push_back((HighsInt) (n_x + d));
            a_value.push_back((double) block_bytes / UNIT);
        }
    }
    a_start.push_back((HighsInt) a_index.size());
    lp.num_row_ = (HighsInt) row_lower.size();
    lp.row_lower_ = row_lower;
    lp.row_upper_ = row_upper;
    lp.a_matrix_.format_ = MatrixFormat::kRowwise;
    lp.a_matrix_.num_row_ = lp.num_row_;
    lp.a_matrix_.num_col_ = lp.num_col_;
    lp.a_matrix_.start_ = a_start;
    lp.a_matrix_.index_ = a_index;
    lp.a_matrix_.value_ = a_value;
    lp.sense_ = ObjSense::kMinimize;

    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("time_limit", in.time_limit_s);
    highs.setOptionValue("mip_rel_gap", 1e-4);
    if (highs.passModel(model) != HighsStatus::kOk) {
        r.error = "HiGHS rejected the model";
        return r;
    }
    const HighsStatus st = highs.run();
    const HighsModelStatus ms = highs.getModelStatus();
    r.t_solve_s = (ggml_time_us() - t0) * 1e-6;
    if (st == HighsStatus::kError || ms == HighsModelStatus::kInfeasible || ms == HighsModelStatus::kUnboundedOrInfeasible) {
        r.error = ms == HighsModelStatus::kInfeasible ? "no placement fits the capacities" : "HiGHS failed";
        return r;
    }
    const HighsSolution & sol = highs.getSolution();
    if (!sol.value_valid || (int) sol.col_value.size() < n_cols) {
        r.error = "HiGHS returned no solution within the time limit";
        return r;
    }
    r.optimal      = ms == HighsModelStatus::kOptimal;
    r.objective_us = highs.getInfo().objective_function_value;

    // ---- the allocation
    r.alloc = in.base;
    for (size_t g = 0; g < groups.size(); g++) {
        size_t best = 0;
        for (size_t k = 1; k < nk; k++) {
            if (sol.col_value[g * nk + k] > sol.col_value[g * nk + best]) {
                best = k;
            }
        }
        for (const size_t i : groups[g].idx) {
            r.alloc.tensor_device[i] = dev_of(best);
        }
    }
    if (move_block) {
        size_t best = 0;
        for (size_t k = 1; k < nd; k++) {
            if (sol.col_value[n_x + k] > sol.col_value[n_x + best]) {
                best = k;
            }
        }
        for (uint32_t il = inv.n_layer; il <= n_layer_all; il++) {
            r.alloc = r.alloc.with_layer_home(inv, il, (int) best);
        }
    }
    r.ok = true;
    return r;
}
#endif
