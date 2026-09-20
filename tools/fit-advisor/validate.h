#pragma once

// validation of one allocation by a real load: allocate everything, run a prompt ubatch and a few generation steps,
// then read what the device actually holds. the difference to the projection is the memory no estimate covers, and
// becomes the margin the search needs

#include "probe.h"

#include <cstdint>
#include <string>
#include <vector>

struct fit_advisor_validate_device {
    std::string name;
    int64_t free_before = 0; // the projection's reading, before anything of this process ran on the device
    int64_t free_after  = 0; // after the prompt and generation steps, everything still allocated
    int64_t free_final  = 0; // after the model and context were freed
    size_t  model   = 0; // buffers actually allocated
    size_t  context = 0;
    size_t  compute = 0;
    size_t  proj_model   = 0; // what the probe projected
    size_t  proj_context = 0;
    size_t  proj_compute = 0;
    size_t  proj_scratch = 0;
    int64_t margin = 0; // the margin the probe applied

    int64_t used()       const { return free_before - free_after; }
    int64_t buffers()    const { return (int64_t) (model + context + compute); }
    int64_t overhead()   const { return used() - buffers(); }                // pool scratch + runtime state
    int64_t unmodelled() const { return overhead() - (int64_t) proj_scratch; } // what only the margin covered
    int64_t projected()  const { return (int64_t) (proj_model + proj_context + proj_compute + proj_scratch); }
};

struct fit_advisor_validate_result {
    bool ok = false;
    std::string error;
    std::vector<fit_advisor_validate_device> devices; // model device order, like the projection
    uint32_t n_prompt_tokens = 0;
    uint32_t n_gen_steps     = 0;
    double   t_load_s = 0;
    double   t_run_s  = 0;

    // the margin the search should use for each device: what the projection missed plus the pad, never negative
    int64_t suggested_margin(size_t device) const;
};

// n_prompt_tokens = 0 chooses two ubatches (or the whole per-slot context if smaller); the prompt is run in n_batch
// pieces through the same decode path as the server, then n_gen_steps single-token steps on every slot
fit_advisor_validate_result fit_advisor_validate(const common_params & params, const fit_advisor_candidate & cand,
                                                 const fit_advisor_projection & proj, uint32_t n_prompt_tokens);

void fit_advisor_validate_print(const fit_advisor_validate_result & vr);
