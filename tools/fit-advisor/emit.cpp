#include "emit.h"

#include "common.h"
#include "log.h"
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

fit_advisor_emit_result fit_advisor_emit_ini(const std::string & path, const std::string & section, const std::string & model_path,
                                             const fit_advisor_candidate & cand, int32_t n_batch_base) {
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
    if (cand.n_ubatch > 0) {
        kv.push_back({ "LLAMA_ARG_UBATCH", std::to_string(cand.n_ubatch) });
        kv.push_back({ "LLAMA_ARG_BATCH", std::to_string(std::max<int32_t>((int32_t) cand.n_ubatch, n_batch_base)) });
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
