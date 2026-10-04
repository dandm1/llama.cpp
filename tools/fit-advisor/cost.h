#pragma once

// cost model: expected time per request for an allocation, from the inventory, the device measurements,
// the memory projection of the candidate (for exact KV bytes per device) and a workload profile
//
// everything here is a pure function over the numbers the other modules produce: no ggml, no llama

#include "allocation.h"
#include "inventory.h"
#include "llama.h"
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
    int32_t  op_offload_min_batch = 0; // batch from which CPU-resident weights are copied to a device and run there; 0 = each device's own
    std::vector<int32_t> op_offload_min_batch_dev; // per allocation device, overrides the scalar where > 0 (empty: none)

    // the threshold that applies to allocation device d (its own measured one when nothing overrides it)
    int32_t offload_min_for(size_t d, int32_t device_own) const {
        if (d < op_offload_min_batch_dev.size() && op_offload_min_batch_dev[d] > 0) {
            return op_offload_min_batch_dev[d];
        }
        return op_offload_min_batch > 0 ? op_offload_min_batch : device_own;
    }
    // share of a layer's experts a prompt ubatch touches: the scheduler copies only those when experts are offloaded.
    // 0 = estimate: routing on real text is skewed, so a fixed 0.6 is used from batch 256 up (measured 0.49 on Qwen3-Next
    // at 1369 tokens, ~0.7 on GLM-5.3 at 512) and the uniform-routing formula below that where it is smaller
    double   expert_coverage = 0;
    bool     pinned_cpu_weights = false; // CPU-resident weights live in the devices' pinned host buffer, copies run at the pinned rate
    bool     use_mtp       = false; // MTP layers are executed (speculative MTP drafting on), otherwise they are not even loaded
    uint32_t mtp_draft_n   = 0;     // draft tokens per step for the per-tensor costs; 0 = a default near the trained depth.
                                    // the cost estimate itself scans the depth and reports its choice in fit_advisor_cost
    uint32_t mtp_trained_depth = 1; // MTP layers in the model: positions up to this depth are drafted as trained
    double   mtp_accept    = 0.8;   // probability that one drafted token is accepted, a heuristic like the token counts
    double   mtp_decay     = 0.85;  // beyond the trained depth a single layer drafts from its own guesses: the acceptance
                                    // of every further position is the base one times this factor (a step, not compounded)
    double   mtp_extra_per_depth_us = 0; // per draft position, what a timed verification row costs beyond the kernels
                                         // the model prices (validation), where no per-depth figure exists
    double   mtp_draft_extra_us = 0;     // a timed draft decode beyond the kernels the model prices for it: the decode's
                                         // launches, synchronisation, embedding and logits transfers (validation), for a
                                         // draft layer on a device not in the list below
    // the same per device the draft layer was validated on (index into the allocation's devices, the CPU last): on a
    // 3090 with a 3060 the overhead was 3.7 ms on the fast card and 9.5 ms on the slow one, for the same kernels.
    // a device without a measurement is priced at the last measured overhead. pricing it optimistically so the search
    // would try the move was tried and sent the walk chasing a move it cannot complete in single steps (the draft
    // layer and the head need room the fast card only has once a trunk layer leaves); that needs an exchange move
    std::vector<double> mtp_draft_extra_by_dev_us;
    double mtp_draft_extra_for(int dev, size_t n_devices) const {
        const size_t i = dev < 0 ? n_devices : (size_t) dev;
        if (i < mtp_draft_extra_by_dev_us.size() && mtp_draft_extra_by_dev_us[i] >= 0) {
            return mtp_draft_extra_by_dev_us[i];
        }
        return mtp_draft_extra_us;
    }
    double   mtp_rollback_us = 0;        // removing a rejected tail from the memory, paid by every step that rejects a
                                         // draft token (validation; a recurrent model restores a state snapshot)
    double   split_extra_us = 0;         // per scheduler split in the generation step, a device change along the layer
                                         // sequence or an op away from its layer's device: what a timed plain step cost
                                         // beyond the kernels and transfers the model prices (validation). a real graph
                                         // loses far more to a split than the link's latency: a crossing measured ~1 ms
    std::vector<uint32_t> mtp_extra_partition;  // the layer partition the per-depth extras were timed on: they hold
                                                // for keys with that partition; other keys get the per-depth scalar
    std::vector<double> mtp_extra_by_depth_us; // [depth] -> the extra for that depth where it was timed, in place of
                                               // depth * mtp_extra_per_depth_us (the growth is not linear in the rows:
                                               // CPU kernels take an odd row count on a slower path)

    // probability that the drafted token at 1-based position k is accepted given every earlier one was
    double mtp_accept_at(uint32_t k) const;
    // tokens produced per generation step and slot at a draft depth: 1 verified token plus the expected accepted drafts
    double tokens_per_step(uint32_t depth) const;
    // the depth the per-tensor costs assume when none was chosen yet
    uint32_t mtp_draft_n_default() const { return mtp_draft_n > 0 ? mtp_draft_n : mtp_trained_depth + 2; }

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

    // breakdown of one decode step, microseconds (with drafting: the verification pass over 1 + draft tokens)
    double step_weights_us  = 0;
    double step_attn_us     = 0;
    double step_overhead_us = 0;
    double step_boundary_us = 0;
    uint32_t n_splits       = 0; // scheduler splits in the generation step: device changes along the layers plus excursions

    // drafting: one run of the MTP layer(s) at the generation batch, the depth the scan chose (0: drafting does not
    // pay on this allocation), the tokens a step yields per slot at that depth, and the rate at every depth tried
    double   t_mtp_draft_us  = 0;
    uint32_t mtp_draft_n     = 0;
    double   tokens_per_step = 1;
    std::vector<double> mtp_depth_tok_s;   // [depth] -> aggregate gen tokens/s, depth 0 = plain steps
    std::vector<double> mtp_depth_step_us; // [depth] -> the step time behind it
    std::vector<double> mtp_depth_rows_us; // [depth] -> the trunk's verification step alone (1 + depth rows), no draft runs

    // objective: lower is better
    double score() const { return t_request_us; }
};

// devices[d] describes allocation device d, devices.back() is the CPU
fit_advisor_cost fit_advisor_cost_estimate(const fit_advisor_inventory & inv, const fit_advisor_allocation & alloc,
                                           const fit_advisor_projection & proj, const fit_advisor_graph_profile & gp,
                                           const std::vector<fit_advisor_cost_device> & devices, const fit_advisor_pair_table & pairs,
                                           const fit_advisor_workload & wl);

// the device's measured weight-streaming rate (best matmul type), bytes/s; 0 without measurements
double fit_advisor_device_rate(const fit_advisor_cost_device & d);

// seconds per weight byte at a batch size, interpolated on the measured curve
double fit_advisor_s_per_byte(const fit_advisor_matmul_rate & r, uint32_t batch);

// microseconds one op over this tensor costs when it lives on dev_idx (DEV_CPU = -1) at a batch size:
//   - matmul weights: bytes on the measured per-byte curve, plus the per-ubatch copy when offloaded
//   - anything else: the device's per-op cost plus the op's activation bytes over the device's memory rate
// home_idx: the device of the tensor's layer, where an offloaded op runs when that device wants it (-1: the first willing one)
// wl: the workload whose offload thresholds apply (nullptr: the devices' own)
constexpr int32_t FIT_ADVISOR_OFFLOAD_NEVER = LLAMA_OP_OFFLOAD_NEVER;
double fit_advisor_tensor_cost_us(const fit_advisor_inventory & inv, const fit_advisor_tensor & t, const fit_advisor_tensor_use & use,
                                  int dev_idx, const std::vector<fit_advisor_cost_device> & devices, uint32_t batch, int home_idx = -1,
                                  const fit_advisor_workload * wl = nullptr);

// microseconds to move the activation of a batch from one allocation device to another (DEV_CPU = -1), through the
// scheduler: half a measured split round trip plus bandwidth for a larger activation
double fit_advisor_hop_us(const fit_advisor_inventory & inv, const std::vector<fit_advisor_cost_device> & devices,
                          const fit_advisor_pair_table & pairs, int from, int to, uint32_t batch);

// microseconds an excursion costs: one op of a layer on home whose weight sits on dev, the layer continuing on home.
// the measured figure for the pair when there is one (it holds the thread pool's wake for a CPU destination), else
// the two hops of the activation
double fit_advisor_excursion_us(const fit_advisor_inventory & inv, const std::vector<fit_advisor_cost_device> & devices,
                                const fit_advisor_pair_table & pairs, int home, int dev, uint32_t batch);

// the allocation device that takes an offloaded op of a layer whose home is home_idx at this batch: the home when it
// wants it, else the first willing one; -1 when none does (the op runs on the CPU)
int fit_advisor_offload_taker(const std::vector<fit_advisor_cost_device> & devices, uint32_t batch, int home_idx, const fit_advisor_workload * wl);

// whether the cost of this tensor on this device rests on measurements (false: a fallback rate was used, or the
// device has no measurements); the search never lets an unknown cost decide a placement
bool fit_advisor_tensor_cost_known(const fit_advisor_inventory & inv, const fit_advisor_tensor & t, const fit_advisor_tensor_use & use,
                                   int dev_idx, const std::vector<fit_advisor_cost_device> & devices);

// microseconds per request attributable to this tensor on this device under the workload
double fit_advisor_tensor_request_us(const fit_advisor_inventory & inv, const fit_advisor_graph_profile & gp, size_t tensor_idx, int dev_idx,
                                     const std::vector<fit_advisor_cost_device> & devices, const fit_advisor_workload & wl, uint32_t n_slots, int home_idx = -1);
