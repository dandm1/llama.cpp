#pragma once

#include "common.h"
#include "llama.h"
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <iomanip>
#include <sstream>
#include <algorithm>

// Structure to track allocation and memory usage of tensors
struct tensor_allocation_info {
    std::string name;               // Name of the tensor
    std::string buffer_type;        // Type of buffer (e.g., CPU, CUDA, Metal)
    size_t size_bytes;              // Size in bytes
    ggml_type type;                 // GGML type of tensor
    std::vector<int64_t> ne;        // Tensor dimensions
    void* data;                     // Pointer to data
    bool is_view;                   // Whether this is a view of another tensor
    std::string view_src;           // If is_view, the source tensor name
};

// Structure to track memory usage per backend/device
struct tensor_memory_usage {
    size_t total_bytes = 0;        // Total memory used in bytes
    size_t num_tensors = 0;        // Number of tensors allocated
    std::vector<tensor_allocation_info> tensor_details; // Details of each tensor
};

// Function to print tensor allocations
void print_tensor_allocations(const std::vector<tensor_allocation_info>& allocations);

// Main function to collect and report model tensors
bool inspect_model_tensors(llama_model* model);

// Function to inspect tensors after model loading
void inspect_tensors_after_loading(llama_model* model, bool inspect_enabled);