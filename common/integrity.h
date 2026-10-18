#pragma once

// periodic integrity check of a loaded model's weights: a digest of every tensor is taken once, then recomputed and
// compared on demand. a mismatch means the bytes in memory changed after loading: a faulty or overheated memory
// chip, a bad copy, or something writing where it should not. the check detects; it does not repair

#include "llama.h"
#include "ggml-backend.h"

#include <cstddef>
#include <map>
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
        // the known-answer matmul on every device and the pattern copy between every pair of devices
        size_t n_compute_fail = 0;
        size_t n_copy_fail    = 0;
        size_t n_copy         = 0;
    };

    // one small fixed workload per device: a matmul whose result digest is taken at init, and a pattern tensor
    // that is copied to every other device and digested on arrival
    struct device_check {
        ggml_backend_dev_t    dev     = nullptr;
        std::string           name;
        ggml_backend_t        backend = nullptr;
        ggml_context *        ctx     = nullptr;
        ggml_backend_buffer_t buf     = nullptr;
        ggml_cgraph *         gf      = nullptr;
        ggml_tensor *         out     = nullptr;
        ggml_tensor *         pattern = nullptr;
        ggml_tensor *         landing = nullptr;
        uint64_t              out_digest = 0;
    };
    std::vector<device_check> devices;
    uint64_t pattern_digest = 0;

    // a telemetry line per accelerator from nvidia-smi, when present: temperature, clocks, power, throttle
    // reason, link state and the change in the PCIe replay counter since the previous pass
    struct telemetry_sample {
        std::string bus_id, device;
        int temp_gpu = 0, temp_mem = -1, clk_sm = 0, clk_mem = 0, link_gen = 0, link_width = 0;
        double power_w = 0;
        unsigned long long throttle = 0;
        long long replays = -1, replays_delta = 0;
    };
    std::map<std::string, long long> replays_prev; // by bus id
    bool telemetry_available = true;
    std::vector<telemetry_sample> telemetry();

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

    ~common_integrity();
    void devices_init();
    void devices_free();
    void devices_verify(report & r);
};
