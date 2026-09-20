#pragma once

// cost model: expected time per request for an allocation, from the inventory, the device measurements,
// the memory projection of the candidate (for exact KV bytes per device) and a workload profile
//
// everything here is a pure function over the numbers the other modules produce: no ggml, no llama

#include "allocation.h"
#include "inventory.h"
#include "measure.h"
#include "probe.h"

#include <string>
#include <vector>

struct fit_advisor_workload {
    uint32_t prompt_tokens = 2048; // fresh prompt tokens per request, cache misses only
    uint32_t gen_tokens    = 512;  // generated tokens per request
    uint32_t concurrency   = 1;    // requests in flight at once
    bool     throughput    = false; // optimise aggregate tokens/s instead of per-request latency
    uint32_t n_ubatch      = 512;  // prompt-processing micro-batch
    bool     use_mtp       = false; // MTP layers are executed (speculative MTP drafting on), otherwise they are not even loaded

    static fit_advisor_workload preset(const std::string & name); // "chat", "rag", "batch", "agent"
};

// per-device facts the cost model needs beyond the measurements
struct fit_advisor_cost_device {
    std::string name;
    const fit_advisor_device_measurements * meas = nullptr;
    int  offload_min_batch = 0;  // batches at or above this run CPU-resident weights on this device via a copy, 0 = never
    bool is_cpu = false;
    uint32_t n_embd = 0;         // activation width, for boundary transfer sizes
};

// transfer rates between devices, indexed [src][dst] over the cost device list (CPU last); missing entries are zero
using fit_advisor_pair_table = std::vector<std::vector<fit_advisor_pair_rate>>;

struct fit_advisor_cost {
    bool   ok = false;
    std::string error;

    double t_gen_step_us   = 0; // one decode step for the active batch
    double t_prompt_us     = 0; // whole prompt of the workload
    double t_request_us    = 0; // prompt + generation, per request
    double gen_tokens_per_s = 0; // aggregate over the active batch
    double prompt_tokens_per_s = 0;

    // breakdown of one decode step, microseconds
    double step_weights_us  = 0;
    double step_attn_us     = 0;
    double step_overhead_us = 0;
    double step_boundary_us = 0;

    // objective: lower is better
    double score() const { return t_request_us; }
};

// devices[d] describes allocation device d, devices.back() is the CPU
fit_advisor_cost fit_advisor_cost_estimate(const fit_advisor_inventory & inv, const fit_advisor_allocation & alloc,
                                           const fit_advisor_projection & proj, const fit_advisor_graph_profile & gp,
                                           const std::vector<fit_advisor_cost_device> & devices, const fit_advisor_pair_table & pairs,
                                           const fit_advisor_workload & wl);

// seconds per weight byte at a batch size, interpolated on the measured curve
double fit_advisor_s_per_byte(const fit_advisor_matmul_rate & r, uint32_t batch);

// microseconds one op over this tensor costs when it lives on dev_idx (DEV_CPU = -1) at a batch size:
//   - matmul weights: bytes on the measured per-byte curve, plus the per-ubatch copy when offloaded
//   - anything else: the device's per-op cost plus the op's activation bytes over the device's memory rate
double fit_advisor_tensor_cost_us(const fit_advisor_inventory & inv, const fit_advisor_tensor & t, const fit_advisor_tensor_use & use,
                                  int dev_idx, const std::vector<fit_advisor_cost_device> & devices, uint32_t batch);

// microseconds per request attributable to this tensor on this device under the workload
double fit_advisor_tensor_request_us(const fit_advisor_inventory & inv, const fit_advisor_graph_profile & gp, size_t tensor_idx, int dev_idx,
                                     const std::vector<fit_advisor_cost_device> & devices, const fit_advisor_workload & wl, uint32_t n_slots);
