#pragma once

// periodic integrity check of a loaded model's weights: a digest of every tensor is taken once, then recomputed and
// compared on demand. a mismatch means the bytes in memory changed after loading: a faulty or overheated memory
// chip, a bad copy, or something writing where it should not. the check detects; it does not repair

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct common_integrity {
    struct entry {
        const ggml_tensor * tensor;
        std::string         device; // buffer name, e.g. CUDA0 or CPU_Mapped
        bool                host;
        size_t              bytes;
        uint64_t            digest;
    };

    struct report {
        size_t n_checked    = 0;
        size_t bytes_checked = 0;
        size_t n_mismatch   = 0;
        double t_ms         = 0;
        std::vector<std::string> mismatches; // "name on device"
    };

    std::vector<entry> entries;
    size_t bytes_device = 0;
    size_t bytes_host   = 0;

    // host tensors are checked in rotation, at most this many bytes per verify() (0 = all of them every time):
    // host memory is large and shared with the model's own reads
    size_t host_bytes_per_pass = (size_t) 4096 * 1024 * 1024;
    size_t host_cursor = 0;

    // digest every tensor of the model; returns the time taken
    double init(const llama_model * model, int n_threads);

    // recompute and compare: every device tensor, and the next slice of host tensors
    report verify(int n_threads);

    // test hook: flip one byte of the named tensor so the next verify() must report it; false if there is no such
    // tensor or its memory cannot be written (a memory-mapped file)
    bool inject(const std::string & name, std::string & error);

    static uint64_t digest_of(const entry & e, int n_threads);
};
