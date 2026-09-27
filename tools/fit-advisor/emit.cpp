#include "emit.h"

#include "common.h"
#include "log.h"
#include "speculative.h"

#include <algorithm>
#include "ggml-backend.h"
#include "preset.h"

#include <fstream>
#include <sstream>
#include <vector>

namespace {

// the INI grammar ends a value at the first '#' or ';', with no escape; refuse rather than write something that reloads wrong
bool ini_safe(const std::string & v) {
    return v.find('#') == std::string::npos && v.find(';') == std::string::npos;
}

// drop an existing [section] block (up to the next header) from the file text
std::string without_section(const std::string & text, const std::string & section) {
    std::istringstream in(text);
    std::string line;
    std::string out;
    bool skipping = false;
    while (std::getline(in, line)) {
        const size_t lb = line.find_first_not_of(" \t");
        if (lb != std::string::npos && line[lb] == '[') {
            const size_t rb = line.find(']', lb);
            std::string name = rb == std::string::npos ? "" : line.substr(lb + 1, rb - lb - 1);
            name.erase(0, name.find_first_not_of(" \t"));
            name.erase(name.find_last_not_of(" \t") + 1);
            skipping = name == section;
        }
        if (!skipping) {
            out += line + "\n";
        }
    }
    return out;
}

} // namespace

std::vector<fit_advisor_passthrough_option> fit_advisor_passthrough(const common_params & params, const fit_advisor_candidate & cand) {
    const common_params defaults = {};
    std::vector<fit_advisor_passthrough_option> ret;
    if (params.cache_type_k != defaults.cache_type_k) {
        ret.push_back({ "LLAMA_ARG_CACHE_TYPE_K", ggml_type_name(params.cache_type_k), std::string("-ctk ") + ggml_type_name(params.cache_type_k) });
    }
    if (params.cache_type_v != defaults.cache_type_v) {
        ret.push_back({ "LLAMA_ARG_CACHE_TYPE_V", ggml_type_name(params.cache_type_v), std::string("-ctv ") + ggml_type_name(params.cache_type_v) });
    }
    if (cand.flash_attn >= 0) {
        const char * v = cand.flash_attn ? "on" : "off";
        ret.push_back({ "LLAMA_ARG_FLASH_ATTN", v, std::string("-fa ") + v });
    }
    if (params.cpuparams.n_threads != defaults.cpuparams.n_threads && params.cpuparams.n_threads > 0) {
        const std::string v = std::to_string(params.cpuparams.n_threads);
        ret.push_back({ "LLAMA_ARG_THREADS", v, "-t " + v });
    }
    if (cand.no_kv_offload) {
        ret.push_back({ "LLAMA_ARG_KV_OFFLOAD", "false", "-nkvo" });
    }
    if (params.kv_unified) {
        ret.push_back({ "LLAMA_ARG_KV_UNIFIED", "true", "-kvu" });
    }
    if (params.load_mode != defaults.load_mode) {
        const char * v = params.load_mode == LLAMA_LOAD_MODE_NONE ? "none" : params.load_mode == LLAMA_LOAD_MODE_MMAP ? "mmap"
                       : params.load_mode == LLAMA_LOAD_MODE_MLOCK ? "mlock" : params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK ? "mmap+mlock" : "auto";
        ret.push_back({ "LLAMA_ARG_LOAD_MODE", v, std::string("--load-mode ") + v });
    } else {
        // weights kept on the CPU run from the loader's own memory rather than page-mapped file pages; the loader
        // itself warns that mmap hurts in that case
        bool cpu_weights = false;
        for (const auto & o : cand.overrides) {
            cpu_weights = cpu_weights || o.buft == "CPU";
        }
        if (cpu_weights || cand.n_gpu_layers == 0) {
            ret.push_back({ "LLAMA_ARG_LOAD_MODE", "none", "--load-mode none" });
        }
    }
    if (!params.devices.empty()) {
        std::string v;
        for (ggml_backend_dev_t dev : params.devices) {
            if (dev == nullptr) {
                continue; // the list is NULL-terminated for the C API
            }
            v += (v.empty() ? "" : ",") + std::string(ggml_backend_dev_name(dev));
        }
        ret.push_back({ "LLAMA_ARG_DEVICE", v, "-dev " + v });
    }
    std::vector<common_speculative_type> types;
    for (const auto t : params.speculative.types) {
        if (t != COMMON_SPECULATIVE_TYPE_DRAFT_MTP && t != COMMON_SPECULATIVE_TYPE_NONE) {
            types.push_back(t);
        }
    }
    if (cand.spec_mtp) {
        types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
    }
    if (!types.empty()) {
        std::string v;
        for (const auto t : types) {
            v += (v.empty() ? "" : ",") + common_speculative_type_to_str(t);
        }
        ret.push_back({ "LLAMA_ARG_SPEC_TYPE", v, "--spec-type " + v });
        if (params.speculative.draft.n_max != defaults.speculative.draft.n_max) {
            const std::string n = std::to_string(params.speculative.draft.n_max);
            ret.push_back({ "LLAMA_ARG_SPEC_DRAFT_N_MAX", n, "--spec-draft-n-max " + n });
        }
    }
    return ret;
}

std::string fit_advisor_passthrough_cli(const common_params & params, const fit_advisor_candidate & cand) {
    std::string ret;
    for (const auto & o : fit_advisor_passthrough(params, cand)) {
        ret += " " + o.cli;
    }
    return ret;
}

fit_advisor_emit_result fit_advisor_emit_ini(const std::string & path, const std::string & section, const std::string & model_path,
                                             const fit_advisor_candidate & cand, int32_t n_batch_base,
                                             const std::vector<fit_advisor_passthrough_option> & passthrough) {
    fit_advisor_emit_result r;
    r.path    = path;
    r.section = section;

    common_preset_context ctx(LLAMA_EXAMPLE_SERVER);
    common_preset p;
    p.name = section;

    std::vector<std::pair<std::string, std::string>> kv;
    kv.push_back({ "LLAMA_ARG_MODEL", model_path });
    kv.push_back({ "LLAMA_ARG_FIT", "off" }); // the placement is explicit, the built-in fitter must not second-guess it
    if (cand.n_ctx > 0) {
        kv.push_back({ "LLAMA_ARG_CTX_SIZE", std::to_string(cand.n_ctx) });
    }
    kv.push_back({ "LLAMA_ARG_N_PARALLEL", std::to_string(cand.n_slots) });
    kv.push_back({ "LLAMA_ARG_N_GPU_LAYERS", std::to_string(cand.n_gpu_layers) });
    if (!cand.tensor_split.empty()) {
        std::string ts;
        for (size_t i = 0; i < cand.tensor_split.size(); i++) {
            ts += (i ? "/" : "") + std::to_string((long long) (cand.tensor_split[i] + 0.5f));
        }
        kv.push_back({ "LLAMA_ARG_TENSOR_SPLIT", ts });
    }
    if (!cand.overrides.empty()) {
        kv.push_back({ "LLAMA_ARG_OVERRIDE_TENSOR", cand.overrides_str() });
    }
    if (!cand.layer_devices.empty()) {
        kv.push_back({ "LLAMA_ARG_OVERRIDE_LAYER_DEVICE", cand.layer_devices_cli() });
    }
    if (cand.n_ubatch > 0) {
        kv.push_back({ "LLAMA_ARG_UBATCH", std::to_string(cand.n_ubatch) });
        kv.push_back({ "LLAMA_ARG_BATCH", std::to_string(std::max<int32_t>((int32_t) cand.n_ubatch, n_batch_base)) });
    }
    for (const auto & o : passthrough) {
        kv.push_back({ o.env, o.value });
    }

    try {
        for (const auto & [env, value] : kv) {
            if (!ini_safe(value)) {
                r.error = "value for " + env + " contains '#' or ';', which the INI reader treats as a comment";
                return r;
            }
            p.set_option(ctx, env, value);
        }
    } catch (const std::exception & e) {
        r.error = e.what();
        return r;
    }
    r.ini = p.to_ini();

    // merge into the file
    std::string existing;
    {
        std::ifstream f(path);
        if (f.good()) {
            existing.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        }
    }
    std::string text = without_section(existing, section);
    if (!text.empty() && text.back() != '\n') {
        text += "\n";
    }
    if (!text.empty() && text.substr(text.size() - 2) != "\n\n") {
        text += "\n";
    }
    text += r.ini;
    {
        std::ofstream f(path, std::ios::trunc);
        if (!f.good()) {
            r.error = "cannot write " + path;
            return r;
        }
        f << text;
    }

    // reload and compare: the reader must give back exactly what was written
    try {
        common_preset global;
        common_presets presets = ctx.load_from_ini(path, global);
        const auto it = presets.find(section);
        if (it == presets.end()) {
            r.error = "section [" + section + "] not found after writing";
            return r;
        }
        for (const auto & [env, value] : kv) {
            std::string got;
            if (!it->second.get_option(env, got)) {
                r.error = "option " + env + " missing after reload";
                return r;
            }
            if (got != value) {
                r.error = "option " + env + " reloads as '" + got + "' instead of '" + value + "'";
                return r;
            }
        }
    } catch (const std::exception & e) {
        r.error = std::string("reload failed: ") + e.what();
        return r;
    }

    r.ok = true;
    return r;
}
