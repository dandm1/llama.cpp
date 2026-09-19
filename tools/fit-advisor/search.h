#pragma once

// search over allocations
//
//   1. seed: for each grid cell (layer partition, ubatch, slots) an exact 0/1 knapsack per device by dynamic
//      programming places each tensor on its layer's device or the CPU, maximising the request time saved under the
//      device's capacity; the capacity comes from the memory probe of the resulting allocation and is refined by
//      re-probing until stable
//   2. simulated annealing from the best seed over the full state (tensor placement, partition, ubatch, slots), scored
//      by the complete cost model plus a memory penalty from a probe-calibrated linear model; every improved incumbent
//      is re-probed so the final answer is verified by the real loader

#include "allocation.h"
#include "cost.h"
#include "inventory.h"
#include "probe.h"

#include <string>
#include <vector>

struct fit_advisor_search_options {
    std::vector<uint32_t> ubatch_options = { 512, 1024, 2048 };
    uint32_t n_ctx      = 0;     // 0 = model default
    uint32_t max_slots  = 1;     // from the workload's concurrency
    int      anneal_iters = 20000;
    uint32_t seed       = 42;
};

struct fit_advisor_search_result {
    bool ok = false;
    std::string name;
    fit_advisor_allocation alloc;
    fit_advisor_candidate  cand;
    fit_advisor_projection proj;
    fit_advisor_cost       cost;
    fit_advisor_workload   wl;      // the workload with the chosen ubatch
    int n_probes = 0;
    int n_cells  = 0;
    int n_anneal_accepted = 0;
    double seed_request_us = 0;    // the best DP seed, for reporting what annealing added
};

fit_advisor_search_result fit_advisor_search(const fit_advisor_inventory & inv, fit_advisor_probe & probe,
                                             const std::vector<std::string> & device_bufts,
                                             const fit_advisor_graph_profile & gp,
                                             const std::vector<fit_advisor_cost_device> & cost_devs,
                                             const fit_advisor_pair_table & pairs,
                                             const fit_advisor_workload & wl_base,
                                             const fit_advisor_search_options & opts);
