#pragma once

// feasibility probe: project the memory a candidate configuration needs on every device
// wraps common_get_device_memory_data so the numbers match what the server will see

#include "common.h"
#include "fit.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct fit_advisor_override {
    std::string pattern; // regex over tensor names
    std::string buft;    // buffer type name, e.g. "CPU" or "CUDA1"
};

struct fit_advisor_candidate {
    std::string name;

    int32_t  n_gpu_layers = -1;             // -1: all layers
    uint32_t n_ctx        = 0;              // 0: from the model, total across slots
    uint32_t n_slots      = 1;              // parallel sequences
    std::vector<float> tensor_split;        // empty: default split
    std::vector<fit_advisor_override> overrides;

    // canonical description used for memoization and display
    std::string key() const;

    // the value of an equivalent -ot argument, empty if there are no overrides
    std::string overrides_str() const;
};

struct fit_advisor_device_projection {
    std::string name;
    int64_t total   = 0;
    int64_t free    = 0; // free at the time of the probe
    size_t  model   = 0;
    size_t  context = 0;
    size_t  compute = 0;
    int64_t margin  = 0; // target margin from --fit-target

    int64_t used()           const { return (int64_t) (model + context + compute); }
    int64_t projected_free() const { return free - used(); }
    bool    fits()           const { return projected_free() >= margin; }
};

struct fit_advisor_host_projection {
    int64_t total  = 0; // physical RAM; free is not shown because the CPU backend reports all of it as free
    size_t model   = 0;
    size_t context = 0;
    size_t compute = 0;

    size_t used() const { return model + context + compute; }
};

struct fit_advisor_projection {
    bool        ok = false;
    std::string error;

    std::vector<fit_advisor_device_projection> devices;
    fit_advisor_host_projection                host;

    uint32_t n_ctx_train = 0; // as reported by the model, for showing the effective context of n_ctx == 0
    double   t_s         = 0; // probe wall time in seconds

    bool fits_all() const;
};

struct fit_advisor_probe {
    explicit fit_advisor_probe(const common_params & params);

    // project a candidate, memoized on the candidate key
    const fit_advisor_projection & run(const fit_advisor_candidate & cand);

    // what the built-in fitter would choose for the same base parameters, as a candidate
    // returns the fitter status; the candidate is filled in on success and failure alike
    common_params_fit_status fitter_choice(fit_advisor_candidate & out);

    size_t n_probes = 0; // probes actually executed (memo misses)

private:
    common_params  base;
    ggml_log_level log_level;
    std::map<std::string, fit_advisor_projection> memo;
};
