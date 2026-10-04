#pragma once

// exact placement for one search key: with the layer partition, the ubatch, the slots, the flags and the drafting
// choice fixed, every movable tensor group is assigned to a device by a mixed-integer program, minimising the
// modelled request time under each device's memory capacity. the draft layers and the output head move together
// as one block whose home is a variable too, since moving them needs room that only trunk layers can free.
//
// the objective is the sum of each group's request time on its device, from the same per-tensor cost the search
// used, plus a penalty for every op that runs away from its layer's device (the hop and the measured split cost).
// what the sum cannot express (consecutive excursions sharing one split, the draft depth scan) the caller gets
// back from the full cost model on the solution.

#include "allocation.h"
#include "cost.h"

#include <cstdint>
#include <string>
#include <vector>

struct fit_advisor_solve_input {
    const fit_advisor_inventory              * inv       = nullptr;
    const fit_advisor_graph_profile          * gp        = nullptr;
    const std::vector<fit_advisor_cost_device> * devices = nullptr; // allocation devices, the CPU last
    const fit_advisor_pair_table             * pairs     = nullptr;
    const fit_advisor_workload               * wl        = nullptr;
    const std::vector<std::string>           * device_bufts = nullptr;

    fit_advisor_allocation base;        // the key: homes as the partition gives them, every tensor at its home
    std::vector<int64_t>   capacity;    // per allocation device: bytes the weights may take (free - overheads - margin - pad)
    bool   move_draft_block = false;    // the MTP layers and the output head may change home
    double time_limit_s     = 20;
};

struct fit_advisor_solve_result {
    bool        ok = false;
    std::string error;
    fit_advisor_allocation alloc;
    double objective_us = 0; // the program's objective at the solution, request microseconds
    int    n_groups     = 0;
    int    n_vars       = 0;
    double t_solve_s    = 0;
    bool   optimal      = false; // false: a feasible solution within the time limit, not proven optimal
};

// available when the tool was built with HiGHS (FIT_ADVISOR_HIGHS); otherwise ok = false with an error
bool fit_advisor_solver_available();
fit_advisor_solve_result fit_advisor_solve_placement(const fit_advisor_solve_input & in);
