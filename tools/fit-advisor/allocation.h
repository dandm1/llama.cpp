#pragma once

// an explicit allocation of a model to devices: the unit the search works on
//
//   - layers are assigned as the loader can express them with -ngl and -ts: a leading block on the CPU,
//     then one contiguous block per device in the model's device order; the output layer follows the last
//     device that holds layers. this decides where each layer's KV cache and compute live.
//   - every weight tensor can be placed on any device independently with -ot, which is where the freedom is.
//
// an allocation is turned into loader arguments by to_candidate(), and verify() checks through a no_alloc
// load that the loader put every tensor where the allocation said.

#include "inventory.h"
#include "probe.h"

#include <cstdint>
#include <string>
#include <vector>

struct fit_advisor_allocation {
    static constexpr int DEV_CPU = -1;

    uint32_t n_ctx    = 0; // total across slots, 0 = model default
    uint32_t n_slots  = 1;
    uint32_t n_ubatch = 0; // 0 = the base parameters' ubatch

    // number of layers on each device in device order, the output layer counts as one layer on the last used device
    // the remaining leading layers are on the CPU
    std::vector<uint32_t> layers_per_device;

    // device of each inventory tensor, DEV_CPU or an index into the device list
    std::vector<int> tensor_device;

    // device index for a layer, DEV_CPU for the leading CPU block; il == n_layer_all is the output layer
    int layer_device(uint32_t il, uint32_t n_layer_all) const;

    int32_t n_gpu_layers() const;

    // an allocation with all layers on the devices in the given proportions and every tensor following its layer
    static fit_advisor_allocation from_layer_split(const fit_advisor_inventory & inv, const std::vector<std::string> & device_bufts,
                                                   const std::vector<uint32_t> & layers_per_device, uint32_t n_ctx, uint32_t n_slots);

    // loader arguments that reproduce this allocation
    // device_bufts: buffer type name of each device in device order, e.g. {"CUDA0", "CUDA1"}
    fit_advisor_candidate to_candidate(const fit_advisor_inventory & inv, const std::vector<std::string> & device_bufts, const std::string & name) const;

    // bytes per device (index) and on the CPU (last entry) held by this allocation's weights
    std::vector<size_t> weight_bytes_per_device(const fit_advisor_inventory & inv, size_t n_devices) const;
};

// load the model in no_alloc mode with the candidate's arguments and check every tensor against the allocation
// returns the number of misplaced tensors, logging each one; -1 if the load failed
int fit_advisor_verify_allocation(const common_params & params, const fit_advisor_inventory & inv, const fit_advisor_allocation & alloc,
                                  const std::vector<std::string> & device_bufts, const fit_advisor_candidate & cand);

// device buffer type names in the model's device order, from a projection
std::vector<std::string> fit_advisor_device_bufts(const fit_advisor_projection & proj);
