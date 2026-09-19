#include "probe.h"

#include "ggml-backend.h"
#include "llama.h"
#include "log.h"

#include <cstring>
#include <map>
#include <sstream>
#include <stdexcept>

std::string fit_advisor_candidate::key() const {
    std::ostringstream ss;
    ss << "ngl=" << n_gpu_layers << " ctx=" << n_ctx << " np=" << n_slots;
    if (!tensor_split.empty()) {
        ss << " ts=";
        for (size_t i = 0; i < tensor_split.size(); i++) {
            ss << (i > 0 ? "," : "") << tensor_split[i];
        }
    }
    const std::string ot = overrides_str();
    if (!ot.empty()) {
        ss << " ot=" << ot;
    }
    return ss.str();
}

std::string fit_advisor_candidate::overrides_str() const {
    std::string ret;
    for (const auto & o : overrides) {
        if (!ret.empty()) {
            ret += ",";
        }
        ret += o.pattern + "=" + o.buft;
    }
    return ret;
}

bool fit_advisor_projection::fits_all() const {
    if (!ok) {
        return false;
    }
    for (const auto & d : devices) {
        if (!d.fits()) {
            return false;
        }
    }
    return true;
}

// buffer types addressable by name, the same set -ot accepts
static std::map<std::string, ggml_backend_buffer_type_t> get_buft_by_name() {
    std::map<std::string, ggml_backend_buffer_type_t> ret;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
        if (buft) {
            ret[ggml_backend_buft_name(buft)] = buft;
        }
    }
    return ret;
}

fit_advisor_probe::fit_advisor_probe(const common_params & params) :
    base(params),
    log_level(params.verbosity >= LOG_LEVEL_DEBUG ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR) {
}

const fit_advisor_projection & fit_advisor_probe::run(const fit_advisor_candidate & cand) {
    const std::string key = cand.key();
    auto it = memo.find(key);
    if (it != memo.end()) {
        return it->second;
    }

    fit_advisor_projection proj;
    const int64_t t0 = ggml_time_us();

    // build the parameters exactly as the server would from an equivalent command line
    common_params p = base;
    p.n_gpu_layers = cand.n_gpu_layers;
    p.n_ctx        = cand.n_ctx;
    p.n_parallel   = (int32_t) cand.n_slots;

    std::memset(p.tensor_split, 0, sizeof(p.tensor_split));
    for (size_t i = 0; i < cand.tensor_split.size() && i < llama_max_devices(); i++) {
        p.tensor_split[i] = cand.tensor_split[i];
    }

    // pattern strings must outlive the probe, keep them beside the override array
    std::vector<std::string> patterns;
    patterns.reserve(cand.overrides.size());
    p.tensor_buft_overrides.clear();
    try {
        const auto bufts = get_buft_by_name();
        for (const auto & o : cand.overrides) {
            auto b = bufts.find(o.buft);
            if (b == bufts.end()) {
                throw std::runtime_error("unknown buffer type '" + o.buft + "' in override '" + o.pattern + "'");
            }
            patterns.push_back(o.pattern);
            p.tensor_buft_overrides.push_back({ patterns.back().c_str(), b->second });
        }
        p.tensor_buft_overrides.push_back({ nullptr, nullptr });

        llama_model_params   mparams = common_model_params_to_llama(p);
        llama_context_params cparams = common_context_params_to_llama(p);

        std::vector<ggml_backend_dev_t> devs;
        uint32_t hp_ngl = 0;
        uint32_t hp_nct = 0;
        uint32_t hp_nex = 0;
        const common_device_memory_data_vec dmds = common_get_device_memory_data(
            p.model.path.c_str(), &mparams, &cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);

        // dmds has one entry per device followed by the host
        proj.n_ctx_train = hp_nct;
        for (size_t id = 0; id < devs.size(); id++) {
            fit_advisor_device_projection d;
            d.name    = std::string(ggml_backend_dev_name(devs[id])) + " (" + ggml_backend_dev_description(devs[id]) + ")";
            d.total   = dmds[id].total;
            d.free    = dmds[id].free;
            d.model   = dmds[id].model;
            d.context = dmds[id].context;
            d.compute = dmds[id].compute;
            d.margin  = id < base.fit_params_target.size() ? (int64_t) base.fit_params_target[id] : 0;
            proj.devices.push_back(d);
        }
        proj.host.total   = dmds.back().total;
        proj.host.model   = dmds.back().model;
        proj.host.context = dmds.back().context;
        proj.host.compute = dmds.back().compute;
        proj.ok = true;
    } catch (const std::exception & e) {
        proj.ok    = false;
        proj.error = e.what();
    }

    proj.t_s = (ggml_time_us() - t0) * 1e-6;
    n_probes++;

    return memo.emplace(key, std::move(proj)).first->second;
}

common_params_fit_status fit_advisor_probe::fitter_choice(fit_advisor_candidate & out) {
    common_params p = base;

    llama_model_params   mparams = common_model_params_to_llama(p);
    llama_context_params cparams = common_context_params_to_llama(p);

    const common_params_fit_status status = common_fit_params(p.model.path.c_str(), &mparams, &cparams,
        p.tensor_split, p.tensor_buft_overrides.data(), p.fit_params_target.data(), p.fit_params_min_ctx,
        nullptr, log_level);

    out = {};
    out.name         = "fit";
    out.n_gpu_layers = mparams.n_gpu_layers;
    out.n_ctx        = cparams.n_ctx;
    out.n_slots      = (uint32_t) p.n_parallel;

    if (mparams.tensor_split) {
        size_t nd = llama_max_devices();
        while (nd > 0 && mparams.tensor_split[nd - 1] == 0.0f) {
            nd--;
        }
        for (size_t i = 0; i < nd; i++) {
            out.tensor_split.push_back(mparams.tensor_split[i]);
        }
    }
    if (mparams.tensor_buft_overrides) {
        for (const auto * o = mparams.tensor_buft_overrides; o->pattern != nullptr; o++) {
            out.overrides.push_back({ o->pattern, ggml_backend_buft_name(o->buft) });
        }
    }
    return status;
}
