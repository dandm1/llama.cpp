#pragma once

// search over allocations
//
// a memory model is built from a few probes up front (context per layer, the compute of a card that runs layers or
// only holds stores, the copies a card holds for offloaded weights, scratch). then for every key (layer partition,
// ubatch, slots, drafting) the placement of every movable group and the draft block is solved exactly as a
// mixed-integer program for the modelled request time, the full cost model prices the solution, and the best key
// wins. the partition is refined around it, the solution polished under the full cost model, and the chosen
// allocation is the one probed through the real loader, re-solved if the probe finds a card over its margin.

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
    int8_t base_flash_attn    = -1;   // the user's -fa
    bool   base_no_kv_offload = false;
    std::vector<std::vector<uint32_t>> extra_partitions; // layers per device to seed from besides the generated ones (the fitter's split)
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
    int n_solves = 0;
    double seed_request_us = 0;    // the best DP seed, for reporting what annealing added
};

fit_advisor_search_result fit_advisor_search(const fit_advisor_inventory & inv, fit_advisor_probe & probe,
                                             const std::vector<std::string> & device_bufts,
                                             const fit_advisor_graph_profile & gp,
                                             const std::vector<fit_advisor_cost_device> & cost_devs,
                                             const fit_advisor_pair_table & pairs,
                                             const fit_advisor_workload & wl_base,
                                             const fit_advisor_search_options & opts);
