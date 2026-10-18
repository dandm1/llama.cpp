#pragma once

#include "common.cuh"

// xor over the words of [data, data + n) of ggml_hash64_word(word, idx) with idx counted from data, into *d_acc
// (which the caller zeroes); the formula mirrors ggml_hash64_* in ggml-impl.h so host and device digests agree
void ggml_cuda_hash_bytes(const void * data, size_t n, unsigned long long * d_acc, cudaStream_t stream);
