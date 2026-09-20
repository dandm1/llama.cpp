#include "validate.h"

#include "common.h"
#include "ggml-backend.h"
#include "llama.h"
#include "../../src/llama-ext.h"
#include "log.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

int64_t fit_advisor_validate_result::suggested_margin(size_t device) const {
    if (device >= devices.size()) {
        return -1;
    }
    return std::max<int64_t>(0, devices[device].unmodelled()) + FIT_ADVISOR_MARGIN_PAD;
}

// index of the projection device that belongs to this backend device, or -1
static int find_device(const std::vector<fit_advisor_validate_device> & devs, ggml_backend_dev_t dev) {
    const std::string prefix = std::string(ggml_backend_dev_name(dev)) + " (";
    for (size_t i = 0; i < devs.size(); i++) {
        if (devs[i].name.rfind(prefix, 0) == 0) {
            return (int) i;
        }
    }
    return -1;
}

static void read_free(std::vector<fit_advisor_validate_device> & devs, int64_t fit_advisor_validate_device::* field) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const int id = find_device(devs, dev);
        if (id < 0) {
            continue;
        }
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        devs[id].*field = (int64_t) free;
    }
}

fit_advisor_validate_result fit_advisor_validate(const common_params & params, const fit_advisor_candidate & cand,
                                                 const fit_advisor_projection & proj, uint32_t n_prompt_tokens) {
    fit_advisor_validate_result vr;
    if (!proj.ok) {
        vr.error = "projection failed: " + proj.error;
        return vr;
    }
    for (const auto & pd : proj.devices) {
        fit_advisor_validate_device d;
        d.name         = pd.name;
        d.free_before  = pd.free;
        d.proj_model   = pd.model;
        d.proj_context = pd.context;
        d.proj_compute = pd.compute;
        d.proj_scratch = pd.scratch;
        d.margin       = pd.margin;
        vr.devices.push_back(d);
    }

    common_params p = params;
    std::vector<std::string> patterns;
    if (!fit_advisor_apply_candidate(p, cand, patterns, vr.error)) {
        return vr;
    }

    const llama_model_params   mparams = common_model_params_to_llama(p);
    const llama_context_params cparams = common_context_params_to_llama(p);

    LOG_INF("%s: loading %s for real (ngl %d, ctx %u, slots %u, ubatch %d) ...\n", __func__, cand.name.c_str(),
        p.n_gpu_layers, p.n_ctx, cand.n_slots, p.n_ubatch);
    const int64_t t0 = ggml_time_us();

    llama_model * model = llama_model_load_from_file(p.model.path.c_str(), mparams);
    if (model == nullptr) {
        vr.error = "model load failed";
        return vr;
    }
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        llama_model_free(model);
        vr.error = "context creation failed (out of memory?)";
        return vr;
    }
    vr.t_load_s = (ggml_time_us() - t0) * 1e-6;
    LOG_INF("%s: loaded in %.1f s\n", __func__, vr.t_load_s);

    // the workload: a prompt on slot 0 through the batch path, then a few single-token steps on every slot so that
    // graph capture and the generation kernels are exercised too
    const uint32_t n_slots   = std::max<uint32_t>(1, cand.n_slots);
    const uint32_t n_ctx_seq = llama_n_ctx_seq(ctx);
    const uint32_t n_batch   = llama_n_batch(ctx);
    const uint32_t n_ubatch  = llama_n_ubatch(ctx);
    vr.n_gen_steps = 6;
    if (n_prompt_tokens == 0) {
        n_prompt_tokens = std::max<uint32_t>(2 * n_ubatch, 1024);
    }
    if (n_ctx_seq > vr.n_gen_steps + 1) {
        n_prompt_tokens = std::min(n_prompt_tokens, n_ctx_seq - vr.n_gen_steps - 1);
    }
    vr.n_prompt_tokens = n_prompt_tokens;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    auto token_at = [&](uint32_t i) -> llama_token {
        return (llama_token) ((i * 7919u + 17u) % (uint32_t) n_vocab);
    };

    const int64_t t1 = ggml_time_us();
    llama_batch batch = llama_batch_init((int32_t) n_batch, 0, (int32_t) n_slots);
    bool decode_ok = true;

    LOG_INF("%s: prompt of %u tokens in batches of %u ...\n", __func__, n_prompt_tokens, n_batch);
    for (uint32_t pos = 0; pos < n_prompt_tokens && decode_ok; pos += n_batch) {
        const uint32_t n = std::min(n_batch, n_prompt_tokens - pos);
        common_batch_clear(batch);
        for (uint32_t i = 0; i < n; i++) {
            common_batch_add(batch, token_at(pos + i), (llama_pos) (pos + i), { 0 }, i + 1 == n);
        }
        const int ret = llama_decode(ctx, batch);
        if (ret != 0) {
            vr.error = "prompt decode failed with " + std::to_string(ret) + " at position " + std::to_string(pos);
            decode_ok = false;
        }
        if (decode_ok && ((pos / n_batch) % 8 == 7 || pos + n >= n_prompt_tokens)) {
            LOG_INF("%s:   %u / %u tokens, %.1f s\n", __func__, pos + n, n_prompt_tokens, (ggml_time_us() - t1) * 1e-6);
        }
    }

    for (uint32_t step = 0; step < vr.n_gen_steps && decode_ok; step++) {
        common_batch_clear(batch);
        for (uint32_t s = 0; s < n_slots; s++) {
            const llama_pos pos = s == 0 ? (llama_pos) (n_prompt_tokens + step) : (llama_pos) step;
            common_batch_add(batch, token_at(1000 + step * n_slots + s), pos, { (llama_seq_id) s }, true);
        }
        const int ret = llama_decode(ctx, batch);
        if (ret != 0) {
            vr.error = "generation decode failed with " + std::to_string(ret) + " at step " + std::to_string(step);
            decode_ok = false;
        }
    }
    llama_synchronize(ctx);
    vr.t_run_s = (ggml_time_us() - t1) * 1e-6;

    // what the device holds now: every buffer plus the pool at its high-water mark plus the runtime's own state
    read_free(vr.devices, &fit_advisor_validate_device::free_after);

    for (const auto & [buft, mb] : llama_get_memory_breakdown(ctx)) {
        if (ggml_backend_buft_is_host(buft)) {
            continue;
        }
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        const int id = dev ? find_device(vr.devices, dev) : -1;
        if (id < 0) {
            continue;
        }
        vr.devices[id].model   += mb.model;
        vr.devices[id].context += mb.context;
        vr.devices[id].compute += mb.compute;
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    read_free(vr.devices, &fit_advisor_validate_device::free_final);

    vr.ok = decode_ok;
    return vr;
}

void fit_advisor_validate_print(const fit_advisor_validate_result & vr) {
    constexpr double MiB = 1024.0 * 1024.0;
    if (!vr.ok) {
        printf("\nvalidation failed: %s\n", vr.error.c_str());
        fflush(stdout);
        return;
    }
    printf("\nvalidation by a real load: %u prompt tokens + %u generation steps, load %.1f s, run %.1f s\n",
        vr.n_prompt_tokens, vr.n_gen_steps, vr.t_load_s, vr.t_run_s);
    printf("%-34s %9s %9s %9s %9s %9s %9s %9s %9s %9s\n",
        "device", "projected", "actual", "buffers", "overhead", "scratch", "unmodel.", "margin", "suggest", "leak");
    for (size_t d = 0; d < vr.devices.size(); d++) {
        const auto & v = vr.devices[d];
        printf("%-34.34s %9.0f %9.0f %9.0f %9.0f %9.0f %9.0f %9.0f %9.0f %9.0f%s\n", v.name.c_str(),
            v.projected() / MiB, v.used() / MiB, v.buffers() / MiB, v.overhead() / MiB, v.proj_scratch / MiB,
            v.unmodelled() / MiB, v.margin / MiB, vr.suggested_margin(d) / MiB, (v.free_before - v.free_final) / MiB,
            v.unmodelled() > v.margin ? "  OVER the margin" : "");
        if (v.model != v.proj_model || v.context != v.proj_context || v.compute != v.proj_compute) {
            printf("%-34s buffers differ from the projection: model %+.0f, context %+.0f, compute %+.0f MiB\n", "",
                ((double) v.model - (double) v.proj_model) / MiB, ((double) v.context - (double) v.proj_context) / MiB,
                ((double) v.compute - (double) v.proj_compute) / MiB);
        }
    }
    printf("[MiB] projected: model+ctx+compute+scratch from the probe; actual: free before minus free after the run;\n");
    printf("buffers: model+context+compute really allocated; overhead: actual - buffers (pool scratch + runtime state);\n");
    printf("unmodel.: overhead - projected scratch, what only the margin covered; suggest: unmodel. + %.0f MiB pad;\n",
        FIT_ADVISOR_MARGIN_PAD / MiB);
    printf("leak: memory not returned after freeing the model (kernel modules stay loaded, that is expected)\n");
    fflush(stdout);
}
