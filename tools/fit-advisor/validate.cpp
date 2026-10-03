#include "validate.h"

#include "common.h"
#include "ggml-backend.h"
#include "llama.h"
#include "../../src/llama-ext.h"
#include "log.h"
#include "speculative.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

// the router's chosen experts are graph tensors named ffn_moe_topk-<il> of shape [n_used, n_tokens]; the context's
// evaluation callback lets the validation read them and count how many distinct experts each ubatch touched
struct coverage_probe {
    uint32_t n_expert = 0;
    uint32_t n_ubatch = 0;
    double   sum_share = 0;
    uint32_t n_samples = 0;
    std::vector<char> seen;
};

bool coverage_cb(struct ggml_tensor * t, bool ask, void * ud) {
    coverage_probe & p = *(coverage_probe *) ud;
    const bool wanted = std::strncmp(ggml_get_name(t), "ffn_moe_topk-", 13) == 0 && t->type == GGML_TYPE_I32;
    if (ask) {
        return wanted;
    }
    if (!wanted || t->ne[1] < 32 || (uint32_t) t->ne[1] != p.n_ubatch) {
        return true; // generation steps and partial ubatches are not what the copies are priced on
    }
    // the ids are a strided view of the argsort output (the first n_used of n_expert columns): read the view's bytes
    // and walk it with its strides
    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
    p.seen.assign(p.n_expert, 0);
    uint32_t distinct = 0;
    for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
        for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
            const int32_t id = *(const int32_t *) (buf.data() + i1 * t->nb[1] + i0 * t->nb[0]);
            if (id >= 0 && (uint32_t) id < p.n_expert && !p.seen[id]) {
                p.seen[id] = 1;
                distinct++;
            }
        }
    }
    p.sum_share += (double) distinct / p.n_expert;
    p.n_samples++;
    return true;
}

// English prose for the validation prompt: routing on real text is skewed in a way random tokens are not
const char * builtin_prompt_text =
    "The city council met on Tuesday evening to discuss the proposed changes to the bus network, which would replace "
    "three of the older routes with a single loop serving the hospital, the university and the retail park. Residents of "
    "the eastern suburbs argued that the loop would add twenty minutes to their journey into the centre, while the "
    "operator pointed to falling passenger numbers and the cost of maintaining the current fleet. After two hours of "
    "debate the council agreed to commission an independent survey of travel patterns before taking a decision. "
    "In other business, the planning committee approved an extension to the primary school, rejected an application "
    "for a drive-through restaurant on the ring road, and noted a report on flood defences that recommended raising "
    "the embankment along the river by half a metre over the next five years. The meeting closed at ten past nine. ";

std::vector<llama_token> validation_tokens(const llama_vocab * vocab, const std::string & file, uint32_t n_tokens, std::string & source) {
    std::string text;
    if (!file.empty()) {
        std::ifstream f(file);
        if (f.good()) {
            std::stringstream ss;
            ss << f.rdbuf();
            text = ss.str();
            source = file;
        } else {
            LOG_WRN("%s: cannot read %s, using the built-in text\n", __func__, file.c_str());
        }
    }
    if (text.empty()) {
        text   = builtin_prompt_text;
        source = "built-in paragraph";
    }
    // repeat the text until it tokenizes to enough tokens
    std::vector<llama_token> toks;
    std::string repeated = text;
    for (int k = 0; k < 16; k++) {
        toks = common_tokenize(vocab, repeated, true);
        if (toks.size() >= n_tokens) {
            break;
        }
        repeated += " " + text;
    }
    if (toks.size() < n_tokens) {
        while (toks.size() < n_tokens) {
            toks.push_back(toks[toks.size() % std::max<size_t>(1, toks.size() / 2)]);
        }
    }
    toks.resize(n_tokens);
    return toks;
}

} // namespace

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
    llama_context_params cparams = common_context_params_to_llama(p);

    coverage_probe probe;
    cparams.cb_eval           = coverage_cb;
    cparams.cb_eval_user_data = &probe;

    LOG_INF("%s: loading %s for real (ngl %d, ctx %u, slots %u, ubatch %d) ...\n", __func__, cand.name.c_str(),
        p.n_gpu_layers, p.n_ctx, cand.n_slots, p.n_ubatch);
    const int64_t t0 = ggml_time_us();

    llama_model * model = llama_model_load_from_file(p.model.path.c_str(), mparams);
    if (model == nullptr) {
        vr.error = "model load failed";
        return vr;
    }
    probe.n_expert = (uint32_t) std::max(0, llama_model_n_expert(model));
    probe.n_ubatch = (uint32_t) p.n_ubatch;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        llama_model_free(model);
        vr.error = "context creation failed (out of memory?)";
        return vr;
    }
    // the MTP draft context on the same weights, as the server creates it; its buffers count, its kernels are not run
    llama_context * ctx_mtp = nullptr;
    if (std::find(p.speculative.types.begin(), p.speculative.types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != p.speculative.types.end()) {
        common_params p_dft = common_base_params_to_speculative(p);
        llama_context_params cparams_dft = common_context_params_to_llama(p_dft);
        cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cparams_dft.n_rs_seq = 0;
        cparams_dft.n_ctx    = cparams.n_ctx;
        ctx_mtp = llama_init_from_model(model, cparams_dft);
        if (ctx_mtp == nullptr) {
            LOG_WRN("%s: MTP draft context creation failed, validating without it\n", __func__);
        }
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
    std::string prompt_source;
    const std::vector<llama_token> prompt = validation_tokens(vocab, params.fit_advisor_validate_prompt, n_prompt_tokens + vr.n_gen_steps * n_slots + 16, prompt_source);
    probe.n_ubatch = n_ubatch;
    auto token_at = [&](uint32_t i) -> llama_token {
        return i < prompt.size() ? prompt[i] : (llama_token) ((i * 7919u + 17u) % (uint32_t) n_vocab);
    };

    const int64_t t1 = ggml_time_us();
    llama_batch batch = llama_batch_init((int32_t) n_batch, 0, (int32_t) n_slots);
    bool decode_ok = true;

    // every prompt token asks for logits: the reserve sizes the compute buffer and the scratch estimate for an
    // output matmul over the whole ubatch, so the validation has to exercise that path too
    LOG_INF("%s: prompt of %u tokens (%s) in batches of %u, logits for every token ...\n", __func__, n_prompt_tokens, prompt_source.c_str(), n_batch);
    for (uint32_t pos = 0; pos < n_prompt_tokens && decode_ok; pos += n_batch) {
        const uint32_t n = std::min(n_batch, n_prompt_tokens - pos);
        common_batch_clear(batch);
        for (uint32_t i = 0; i < n; i++) {
            common_batch_add(batch, token_at(pos + i), (llama_pos) (pos + i), { 0 }, true);
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

    // drafting: time what the cost model prices from kernels alone. a verification step of 1 + d rows and a decode of
    // the draft context each carry launch, synchronisation and host work the model cannot see, so they are measured
    // here, at steady state, and the search is told the difference. the KV is rolled back after every step
    if (ctx_mtp && decode_ok) {
        llama_memory_t mem = llama_get_memory(ctx);
        auto pos_of = [&](uint32_t s) -> llama_pos {
            return s == 0 ? (llama_pos) (n_prompt_tokens + vr.n_gen_steps) : (llama_pos) vr.n_gen_steps;
        };
        auto time_decode = [&](llama_context * c, llama_batch & b) -> double {
            const int64_t t = ggml_time_us();
            if (llama_decode(c, b) != 0) {
                return -1;
            }
            for (int32_t i = 0; i < b.n_tokens; i++) {
                if (b.logits[i]) {
                    llama_get_logits_ith(c, i); // the sampler reads every output row, which waits for the graph
                }
            }
            llama_synchronize(c);
            return (double) (ggml_time_us() - t);
        };
        auto median_after_warmup = [](std::vector<double> ts) -> double {
            if (ts.size() > 1) {
                ts.erase(ts.begin());
            }
            if (ts.empty()) {
                return 0;
            }
            std::sort(ts.begin(), ts.end());
            return ts[ts.size() / 2];
        };
        constexpr int reps = 5;
        const uint32_t d_max = std::min<uint32_t>(8, std::max<uint32_t>(1, n_batch / n_slots) - 1);
        vr.t_verify_us.assign(d_max + 1, 0.0);
        for (uint32_t d = 0; d <= d_max && decode_ok; d++) {
            std::vector<double> ts;
            for (int r = 0; r < reps; r++) {
                common_batch_clear(batch);
                for (uint32_t s = 0; s < n_slots; s++) {
                    for (uint32_t k = 0; k <= d; k++) {
                        common_batch_add(batch, token_at(3000 + (r * n_slots + s) * 16 + k), pos_of(s) + (llama_pos) k, { (llama_seq_id) s }, true);
                    }
                }
                const double t = time_decode(ctx, batch);
                for (uint32_t s = 0; s < n_slots; s++) {
                    llama_memory_seq_rm(mem, (llama_seq_id) s, pos_of(s), -1);
                }
                if (t < 0) {
                    decode_ok = false;
                    vr.error = "timed decode failed";
                    break;
                }
                ts.push_back(t);
            }
            const double t_med = median_after_warmup(ts);
            if (d == 0) {
                vr.t_step_plain_us = t_med;
            } else {
                vr.t_verify_us[d] = t_med;
            }
        }
        if (decode_ok) {
            // the draft context takes the token and the trunk's hidden row for it; the values do not matter to the kernels
            const int32_t n_embd = llama_model_n_embd(model);
            llama_batch bd = llama_batch_init((int32_t) n_slots, n_embd, (int32_t) n_slots);
            bd.token = (llama_token *) malloc(sizeof(llama_token) * n_slots);
            std::fill(bd.embd, bd.embd + (size_t) n_slots * n_embd, 0.01f);
            std::vector<double> ts;
            for (int r = 0; r < reps + 1; r++) {
                common_batch_clear(bd);
                for (uint32_t s = 0; s < n_slots; s++) {
                    common_batch_add(bd, token_at(4000 + r * n_slots + s), (llama_pos) r, { (llama_seq_id) s }, true);
                }
                const double t = time_decode(ctx_mtp, bd);
                if (t >= 0) {
                    ts.push_back(t);
                }
            }
            vr.t_draft_us = median_after_warmup(ts);
            free(bd.token);
            bd.token = nullptr;
            llama_batch_free(bd);
        }
        std::string by_depth;
        for (uint32_t d = 1; d <= d_max; d++) {
            by_depth += string_format(" d=%u %.0f", d, vr.t_verify_us[d]);
        }
        LOG_INF("%s: timed drafting: plain step %.0f us, draft decode %.0f us, verification of 1+d rows:%s us\n", __func__,
            vr.t_step_plain_us, vr.t_draft_us, by_depth.c_str());
    }
    vr.t_run_s = (ggml_time_us() - t1) * 1e-6;
    if (probe.n_samples > 0) {
        vr.expert_coverage  = probe.sum_share / probe.n_samples;
        vr.coverage_samples = probe.n_samples;
        vr.coverage_ubatch  = probe.n_ubatch;
    }

    // what the device holds now: every buffer plus the pool at its high-water mark plus the runtime's own state
    read_free(vr.devices, &fit_advisor_validate_device::free_after);

    auto add_breakdown = [&](llama_context * c, bool with_model) {
        for (const auto & [buft, mb] : llama_get_memory_breakdown(c)) {
            if (ggml_backend_buft_is_host(buft)) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
            const int id = dev ? find_device(vr.devices, dev) : -1;
            if (id < 0) {
                continue;
            }
            if (with_model) {
                vr.devices[id].model += mb.model;
            }
            vr.devices[id].context += mb.context;
            vr.devices[id].compute += mb.compute;
        }
    };
    add_breakdown(ctx, true);
    if (ctx_mtp) {
        add_breakdown(ctx_mtp, false); // shares the weights
    }

    llama_batch_free(batch);
    if (ctx_mtp) {
        llama_free(ctx_mtp);
    }
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
    if (vr.t_draft_us > 0) {
        printf("drafting timed: plain step %.1f ms, draft decode %.2f ms per token, verification step by draft depth:", vr.t_step_plain_us * 1e-3, vr.t_draft_us * 1e-3);
        for (size_t d = 1; d < vr.t_verify_us.size(); d++) {
            printf(" %zu: %.1f", d, vr.t_verify_us[d] * 1e-3);
        }
        printf(" ms\n");
    }
    if (vr.coverage_samples > 0) {
        printf("expert coverage: a %u-token ubatch routes to %.0f%% of a layer's experts on average (%u layer samples); "
               "that share of each CPU-resident expert stack is copied per ubatch\n",
            vr.coverage_ubatch, 100.0 * vr.expert_coverage, vr.coverage_samples);
    }
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
