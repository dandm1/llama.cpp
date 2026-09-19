#pragma once

// model inventory: what tensors a GGUF contains and how they group into layers
// read from the GGUF metadata only, no llama_model is constructed

#include "ggml.h"

#include <cstdint>
#include <string>
#include <vector>

enum fit_advisor_tensor_kind {
    FIT_ADVISOR_TENSOR_ATTN,        // attention weights of a layer
    FIT_ADVISOR_TENSOR_FFN,         // dense FFN weights of a layer (incl. shared experts)
    FIT_ADVISOR_TENSOR_FFN_EXPS,    // MoE expert weights of a layer
    FIT_ADVISOR_TENSOR_LAYER_OTHER, // norms, biases, recurrent state and anything else inside a layer
    FIT_ADVISOR_TENSOR_TOKEN_EMBD,  // token_embd.weight
    FIT_ADVISOR_TENSOR_OUTPUT,      // output.weight
    FIT_ADVISOR_TENSOR_GLOBAL,      // everything else outside the layers (output_norm, rope_freqs, ...)
};

const char * fit_advisor_tensor_kind_name(fit_advisor_tensor_kind kind);

struct fit_advisor_tensor {
    std::string             name;
    ggml_type               type;
    size_t                  nbytes;
    int32_t                 layer; // -1 when the tensor is not part of a layer
    fit_advisor_tensor_kind kind;
};

struct fit_advisor_layer {
    size_t attn     = 0;
    size_t ffn      = 0;
    size_t ffn_exps = 0;
    size_t other    = 0;

    size_t total() const { return attn + ffn + ffn_exps + other; }
    size_t dense() const { return attn + ffn + other; } // everything except MoE experts
};

struct fit_advisor_inventory {
    std::string path;
    std::string arch;

    uint32_t n_layer       = 0; // regular transformer layers
    uint32_t n_layer_nextn = 0; // MTP layers stored after the regular ones (blk.n_layer, ...)
    uint32_t n_expert      = 0;
    uint32_t n_ctx_train   = 0;
    uint32_t n_split       = 1; // number of GGUF shards

    std::vector<fit_advisor_tensor> tensors;
    std::vector<fit_advisor_layer>  layers; // n_layer + n_layer_nextn entries

    size_t token_embd = 0;
    size_t output     = 0;
    size_t global     = 0;
    size_t total      = 0;

    bool is_moe() const { return n_expert > 0; }

    // bytes of a given kind across the layers [il_begin, il_end)
    size_t layer_bytes(uint32_t il_begin, uint32_t il_end, fit_advisor_tensor_kind kind) const;
};

// throws std::runtime_error on failure
fit_advisor_inventory fit_advisor_inventory_load(const std::string & path);

// log a summary at info level
void fit_advisor_inventory_print(const fit_advisor_inventory & inv);
