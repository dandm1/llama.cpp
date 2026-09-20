#pragma once

// per-device performance measurements, cached by a hardware/build fingerprint
//
// each number is a microbenchmark of one ggml op on one backend, not a model benchmark:
//   - matmul: bytes/s of weights streamed at batch 1 (token generation), GFLOPS at a large batch (prompt processing),
//             and the fixed per-op overhead, per weight type
//   - attention: KV bytes/s at batch 1 with and without flash attention, per KV type and head size
//   - copy: host<->device bandwidth and small-transfer latency

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct fit_advisor_device_fingerprint {
    std::string name;        // e.g. "CUDA0"
    std::string description; // e.g. "NVIDIA GeForce RTX 4090"
    std::string device_id;   // PCI id if known
    std::string backend;     // backend registry name
    int64_t     total_memory = 0;
    std::string build_commit;
    int         n_threads = 0; // only meaningful for the CPU device

    std::string key() const;
};

struct fit_advisor_matmul_rate {
    bool   supported   = false;
    double bytes_per_s = 0; // weight bytes per second at batch 1, from the slope between two weight sizes
    double overhead_us = 0; // fixed cost per op at batch 1, the intercept
    double gflops_pp   = 0; // at the prompt-processing batch size, using the small weight
    int    n_batch_pp  = 0;

    // seconds per weight byte at three batch sizes, the curve the cost model interpolates on
    double s_per_byte_b1   = 0; // large weight, batch 1
    double s_per_byte_b4   = 0; // large weight, batch 4
    double s_per_byte_bpp  = 0; // small weight, batch n_batch_pp
    size_t bytes_small = 0; // weight sizes used, the large one is chosen to exceed on-chip caches
    size_t bytes_large = 0;

    // the full curve: batch -> seconds per weight byte for the whole batch (large weight up to batch 64, small above)
    std::map<int, double> points;
};

// expert matmul (mul_mat_id) with the model's own routing geometry: a stack of n_expert experts, each token reading
// n_expert_used of them. the routing, gather and sort costs are inside the measurement, so the cost model needs no
// expert arithmetic for a type that has this
struct fit_advisor_moe_rate {
    bool   supported     = false;
    int    n_expert      = 0;
    int    n_expert_used = 0;
    int64_t k = 0, m = 0;          // one expert's matrix, [k, m]
    size_t bytes_total = 0;        // the whole stack
    std::map<int, double> points;  // batch (tokens) -> seconds per stack byte for the whole ubatch
};

struct fit_advisor_attn_rate {
    bool   supported_fa    = false;
    bool   supported_nofa  = false;
    double kv_bytes_per_s_fa   = 0; // K+V bytes read per second at batch 1 with flash attention
    double kv_bytes_per_s_nofa = 0; // same through the KQ / softmax / KQV path
    double us_pp_fa   = 0;          // microseconds for one prompt-processing batch over the same KV
    double us_pp_nofa = 0;
    int    n_kv       = 0;
    int    n_batch_pp = 0;
};

struct fit_advisor_copy_rate {
    double h2d_gb_s     = 0;
    double d2h_gb_s     = 0;
    double latency_us   = 0; // one small host-to-device transfer including synchronization
};

// a transfer between two backends as the scheduler performs it, whatever path the hardware takes
struct fit_advisor_pair_rate {
    double latency_us = 0; // a small tensor, including synchronization
    double gb_s       = 0; // a large tensor

    // cost of a scheduler split: a chain of ops on the source device with an excursion to the destination and back,
    // measured through ggml_backend_sched so it includes the copies, the destination's graph compute and whatever the
    // source loses around the split; microseconds per round trip, for a batch-1 and a prompt-sized activation
    double split_us_b1    = 0;
    double split_us_bpp   = 0;
    size_t split_bytes_bpp = 0; // activation bytes moved in the prompt-sized measurement
    int    split_n_batch_pp = 0;
};

struct fit_advisor_device_measurements {
    fit_advisor_device_fingerprint fingerprint;
    std::string measured_at;

    std::map<std::string, fit_advisor_matmul_rate> matmul; // keyed by ggml type name, e.g. "q4_K"
    std::map<std::string, fit_advisor_moe_rate>    moe;    // keyed by ggml type name; one routing geometry per entry
    std::map<std::string, fit_advisor_attn_rate>   attn;   // keyed by "hd<head size>/<kv type>", e.g. "hd128/f16"
    double op_overhead_us = 0; // fixed cost per graph op inside a graph, from the slope of a chain of tiny ops
    double launch_us      = 0; // cost of one graph compute of a single tiny op: what every scheduler split pays on this device

    // device memory the runtime keeps after the benchmark kernels have run and every buffer is freed: lazily loaded
    // kernel modules and other driver state that no buffer accounts for; -1 = not measured, 0 for the CPU
    int64_t runtime_overhead_bytes = -1;

    bool has_matmul(ggml_type type) const { return matmul.count(ggml_type_name(type)) > 0; }
    bool has_matmul_curve(ggml_type type, const std::vector<int> & batches) const; // measured at every batch in the list
    bool has_moe(ggml_type type, int n_expert, int n_expert_used, const std::vector<int> & batches) const;
    bool has_attn(int head_size, int head_size_v, ggml_type type_kv) const;
    fit_advisor_copy_rate copy;
};

struct fit_advisor_measure_options {
    std::vector<ggml_type> weight_types = { GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_F16 };
    std::vector<ggml_type> kv_types     = { GGML_TYPE_F16, GGML_TYPE_Q8_0 };
    int  head_size   = 128; // K width per head
    int  head_size_v = 0;   // V width per head, 0 = same as K
    int  n_head     = 32;
    int  n_head_kv  = 8;
    int  n_kv          = 16384;
    int  n_batch_pp    = 512; // matmul prompt-processing batch
    std::vector<int> batches = { 1, 4, 16, 64 }; // matmul curve points below the prompt batch, which is always added
    std::vector<ggml_type> moe_types; // expert weight types to measure through mul_mat_id, empty for dense models
    int  n_expert      = 0;
    int  n_expert_used = 0;
    int64_t moe_k = 0;  // one expert's matrix, from the model
    int64_t moe_m = 0;
    int  n_batch_attn  = 128; // attention prompt-processing batch, smaller because the KV is large
    int  n_threads     = 0;   // CPU device only, 0 = default
    bool measure_copy  = true;
    bool verbose       = false;
};

// fingerprint of a device for the current build
fit_advisor_device_fingerprint fit_advisor_fingerprint(ggml_backend_dev_t dev, int n_threads);

// run every measurement on one device; slow, seconds to a minute
fit_advisor_device_measurements fit_advisor_measure_device(ggml_backend_dev_t dev, const fit_advisor_measure_options & opts);

// cache of measurements keyed by fingerprint
struct fit_advisor_measurements {
    std::map<std::string, fit_advisor_device_measurements> devices;
    std::map<std::string, fit_advisor_pair_rate> pairs; // keyed by "<src key>-><dst key>"

    // cached entry for this device covering every type in opts, measuring only what is missing and saving;
    // with force set everything is measured again
    const fit_advisor_device_measurements & ensure(ggml_backend_dev_t dev, const fit_advisor_measure_options & opts, bool force);

    // the entry already cached for this device, or nullptr
    const fit_advisor_device_measurements * find(ggml_backend_dev_t dev, int n_threads) const;

    // transfer rates between every ordered pair of the given devices, measuring the missing ones and saving
    void ensure_pairs(const std::vector<ggml_backend_dev_t> & devs, int n_threads, bool force);
    const fit_advisor_pair_rate * find_pair(ggml_backend_dev_t src, ggml_backend_dev_t dst, int n_threads) const;

    static std::string default_path();
    bool load(const std::string & path);
    bool save(const std::string & path) const;

    std::string to_json() const;
};

// log the measurements of one device at info level
void fit_advisor_measurements_print(const fit_advisor_device_measurements & m);

// smallest batch at which this device runs CPU-resident weights itself via a copy (op offload), 0 if never
int fit_advisor_offload_min_batch(ggml_backend_dev_t dev);
