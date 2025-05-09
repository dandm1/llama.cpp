#include "tensor-inspector.h"
#include "log.h"
#include <algorithm>
#include <iostream>
#include <iomanip>

// Implementation of tensor inspection logic
void inspect_tensors_after_loading(llama_model* model, bool inspect_enabled) {
    if (!inspect_enabled || !model) {
        return;
    }

    printf("\n================ Tensor Inspection ================\n");
    printf("Inspecting model tensors...\n");
    
    // Use the public API to inspect the model tensors
    if (inspect_model_tensors(model)) {
        printf("Tensor inspection completed.\n");
    } else {
        printf("Failed to inspect tensors.\n");
    }
}

// Print tensor allocations with detailed information
void print_tensor_allocations(const std::vector<tensor_allocation_info>& allocations) {
    // Calculate total memory
    size_t total_memory = 0;
    for (const auto& alloc : allocations) {
        total_memory += alloc.size_bytes;
    }
    
    // Group by buffer type
    std::map<std::string, tensor_memory_usage> buffer_type_stats;
    
    // Collect stats
    for (const auto& alloc : allocations) {
        buffer_type_stats[alloc.buffer_type].total_bytes += alloc.size_bytes;
        buffer_type_stats[alloc.buffer_type].num_tensors++;
        buffer_type_stats[alloc.buffer_type].tensor_details.push_back(alloc);
    }
    
    // Print summary
    printf("\n================ Tensor Allocation Summary ================\n");
    printf("Total memory used: %.2f MB across %zu tensors\n", 
           total_memory / (1024.0 * 1024.0), allocations.size());
    printf("\n");
    
    printf("--- Memory usage by backend/buffer type ---\n");
    for (const auto& stat : buffer_type_stats) {
        printf("%-12s: %.2f MB (%zu tensors, %.1f%% of total)\n", 
               stat.first.c_str(), 
               stat.second.total_bytes / (1024.0 * 1024.0),
               stat.second.num_tensors,
               (100.0 * stat.second.total_bytes) / total_memory);
    }
    
    // Print detailed tensor info
    printf("\n--- Top 20 largest tensors ---\n");
    printf("%-40s %-12s %-12s %-20s %-12s\n", 
           "Tensor Name", "Buffer Type", "Size (MB)", "Dimensions", "Data Type");
    printf("--------------------------------------------------------------------------------------\n");
    
    // Sort tensors by size (descending)
    std::vector<tensor_allocation_info> sorted_tensors = allocations;
    std::sort(sorted_tensors.begin(), sorted_tensors.end(), 
              [](const tensor_allocation_info& a, const tensor_allocation_info& b) {
                  return a.size_bytes > b.size_bytes;
              });
    
    // Print top 20 largest tensors
    size_t count = 0;
    for (const auto& tensor : sorted_tensors) {
        if (count++ >= 20) break;
        
        // Format dimensions
        std::stringstream dims;
        dims << "[";
        for (size_t i = 0; i < tensor.ne.size(); i++) {
            dims << tensor.ne[i];
            if (i < tensor.ne.size() - 1) dims << ", ";
        }
        dims << "]";
        
        printf("%-40s %-12s %-12.2f %-20s %-12s\n", 
               tensor.name.c_str(),
               tensor.buffer_type.c_str(), 
               tensor.size_bytes / (1024.0 * 1024.0),
               dims.str().c_str(),
               ggml_type_name(tensor.type));
    }
    
    printf("\nNote: Use tensor override options (-ot/--override-tensor) to change tensor placement.\n");
    printf("Example: --override-tensor \"output*=CPU\" to move tensors starting with 'output' to CPU.\n");
}

// Main function to collect and report model tensors
bool inspect_model_tensors(llama_model* model) {
    if (!model) return false;
    
    // Get basic model information available from the public API
    printf("\n================ Model Information ================\n");

    // Get the model's metadata
    printf("Model metadata:\n");
    int n_kv = llama_model_meta_count(model);
    for (int i = 0; i < n_kv; i++) {
        char key_buffer[128];
        if (llama_model_meta_key_by_index(model, i, key_buffer, sizeof(key_buffer)) > 0) {
            char val_buffer[256];
            if (llama_model_meta_val_str_by_index(model, i, val_buffer, sizeof(val_buffer)) > 0) {
                printf("  %s = %s\n", key_buffer, val_buffer);
            }
        }
    }
    
    // Get memory requirements
    size_t mem_required = llama_model_size(model);
    
    printf("\nMemory usage:\n");
    printf("  Model size: %.2f MB\n", mem_required / (1024.0 * 1024.0));
    
    // Get model parameters
    size_t n_params = llama_model_n_params(model);
    printf("  Number of parameters: %.2f billion\n", n_params / 1e9);
    
    // Get vocabulary information
    const llama_vocab* vocab = llama_model_get_vocab(model);
    printf("  Vocabulary size: %d tokens\n", llama_vocab_n_tokens(vocab));
    
    // Show information about tensor overrides
    printf("\nNote: Use tensor override options (-ot/--override-tensor) to change tensor placement.\n");
    printf("Example: --override-tensor \"*attention*=CPU\" to move attention tensors to CPU.\n");
    
    return true;
}