#include "allocation.h"

#include "common.h"
#include "llama.h"
#include "../../src/llama-ext.h"
#include "log.h"

#include <algorithm>
#include <map>
#include <regex>
#include <set>

int fit_advisor_allocation::layer_device(uint32_t il, uint32_t n_layer_all) const {
    const int32_t ngl = n_gpu_layers();
    const int64_t i_gpu_start = (int64_t) n_layer_all + 1 - ngl;
    if ((int64_t) il < i_gpu_start) {
        return DEV_CPU;
    }
    // walk the per-device blocks
    int64_t off = (int64_t) il - i_gpu_start;
    for (size_t d = 0; d < layers_per_device.size(); d++) {
        if (off < (int64_t) layers_per_device[d]) {
            return (int) d;
        }
        off -= layers_per_device[d];
    }
    return DEV_CPU; // unreachable when il <= n_layer_all
}

int32_t fit_advisor_allocation::n_gpu_layers() const {
    int32_t ret = 0;
    for (const uint32_t n : layers_per_device) {
        ret += (int32_t) n;
    }
    return ret;
}

fit_advisor_allocation fit_advisor_allocation::from_layer_split(const fit_advisor_inventory & inv, const std::vector<std::string> & device_bufts,
                                                                const std::vector<uint32_t> & layers_per_device, uint32_t n_ctx, uint32_t n_slots) {
    fit_advisor_allocation a;
    a.n_ctx   = n_ctx;
    a.n_slots = n_slots;
    a.layers_per_device = layers_per_device;
    a.layers_per_device.resize(device_bufts.size(), 0);

    const uint32_t n_layer_all = inv.n_layer + inv.n_layer_nextn;
    a.tensor_device.resize(inv.tensors.size());
    for (size_t i = 0; i < inv.tensors.size(); i++) {
        const auto & t = inv.tensors[i];
        switch (t.kind) {
            case FIT_ADVISOR_TENSOR_TOKEN_EMBD: a.tensor_device[i] = DEV_CPU; break;                            // input layer is always on the CPU
            case FIT_ADVISOR_TENSOR_OUTPUT:
            case FIT_ADVISOR_TENSOR_GLOBAL:     a.tensor_device[i] = a.layer_device(n_layer_all, n_layer_all); break; // output layer
            default:                            a.tensor_device[i] = a.layer_device((uint32_t) t.layer, n_layer_all); break;
        }
    }
    return a;
}

fit_advisor_candidate fit_advisor_allocation::to_candidate(const fit_advisor_inventory & inv, const std::vector<std::string> & device_bufts, const std::string & name) const {
    fit_advisor_candidate c;
    c.name         = name;
    c.n_ctx        = n_ctx;
    c.n_slots      = n_slots;
    c.n_ubatch     = n_ubatch;
    c.n_gpu_layers = n_gpu_layers();

    // -ts as integer layer counts reproduces the blocks exactly, see get_layer_buft_list in llama-model.cpp
    if (device_bufts.size() > 1) {
        for (size_t d = 0; d < device_bufts.size(); d++) {
            c.tensor_split.push_back(d < layers_per_device.size() ? (float) layers_per_device[d] : 0.0f);
        }
    }

    // -ot for every tensor that does not follow its layer, grouped by (device, name suffix) with a layer alternation
    const uint32_t n_layer_all = inv.n_layer + inv.n_layer_nextn;
    static const std::regex re_layer(R"(^blk\.(\d+)\.(.+)$)");

    std::map<std::pair<int, std::string>, std::set<uint32_t>> groups; // (device, suffix) -> layers
    std::map<int, std::vector<std::string>> singles;                  // device -> exact names outside the layers

    for (size_t i = 0; i < inv.tensors.size() && i < tensor_device.size(); i++) {
        const auto & t = inv.tensors[i];
        int dev_default;
        switch (t.kind) {
            case FIT_ADVISOR_TENSOR_TOKEN_EMBD: dev_default = DEV_CPU; break;
            case FIT_ADVISOR_TENSOR_OUTPUT:
            case FIT_ADVISOR_TENSOR_GLOBAL:     dev_default = layer_device(n_layer_all, n_layer_all); break;
            default:                            dev_default = layer_device((uint32_t) t.layer, n_layer_all); break;
        }
        if (tensor_device[i] == dev_default) {
            continue;
        }
        std::smatch m;
        if (std::regex_match(t.name, m, re_layer)) {
            groups[{ tensor_device[i], m[2] }].insert((uint32_t) std::stoul(m[1]));
        } else {
            singles[tensor_device[i]].push_back(t.name);
        }
    }

    auto buft_name = [&](int dev) -> std::string {
        return dev == DEV_CPU ? "CPU" : device_bufts.at(dev);
    };

    for (const auto & [key, layers] : groups) {
        std::string alt;
        for (const uint32_t il : layers) {
            alt += (alt.empty() ? "" : "|") + std::to_string(il);
        }
        c.overrides.push_back({ R"(^blk\.()" + alt + R"()\.)" + regex_escape(key.second) + "$", buft_name(key.first) });
    }
    for (const auto & [dev, names] : singles) {
        for (const auto & n : names) {
            c.overrides.push_back({ "^" + regex_escape(n) + "$", buft_name(dev) });
        }
    }
    return c;
}

std::vector<size_t> fit_advisor_allocation::weight_bytes_per_device(const fit_advisor_inventory & inv, size_t n_devices) const {
    std::vector<size_t> ret(n_devices + 1, 0);
    for (size_t i = 0; i < inv.tensors.size() && i < tensor_device.size(); i++) {
        const int d = tensor_device[i];
        ret[d == DEV_CPU ? n_devices : (size_t) d] += inv.tensors[i].nbytes;
    }
    return ret;
}

std::vector<std::string> fit_advisor_device_bufts(const fit_advisor_projection & proj) {
    std::vector<std::string> ret;
    for (const auto & d : proj.devices) {
        // the projection names devices as "<name> (<description>)", the buffer type is named after the device
        ret.push_back(d.name.substr(0, d.name.find(" (")));
    }
    return ret;
}

namespace {
struct verify_ud {
    const fit_advisor_inventory * inv;
    const fit_advisor_allocation * alloc;
    const std::vector<std::string> * device_bufts;
    std::map<std::string, size_t> index_by_name;
    int n_bad = 0;
    int n_checked = 0;
};
}

int fit_advisor_verify_allocation(const common_params & params, const fit_advisor_inventory & inv, const fit_advisor_allocation & alloc,
                                  const std::vector<std::string> & device_bufts, const fit_advisor_candidate & cand) {
    // same parameter construction as the probe
    common_params p = params;
    p.n_gpu_layers = cand.n_gpu_layers;
    p.n_ctx        = cand.n_ctx;
    p.n_parallel   = (int32_t) cand.n_slots;
    if (cand.n_ubatch > 0) {
        p.n_ubatch = (int32_t) cand.n_ubatch;
        p.n_batch  = std::max(p.n_batch, p.n_ubatch);
    }
    std::fill(p.tensor_split, p.tensor_split + llama_max_devices(), 0.0f);
    for (size_t i = 0; i < cand.tensor_split.size() && i < llama_max_devices(); i++) {
        p.tensor_split[i] = cand.tensor_split[i];
    }
    std::map<std::string, ggml_backend_buffer_type_t> bufts;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_buffer_type_t b = ggml_backend_dev_buffer_type(ggml_backend_dev_get(i));
        if (b) {
            bufts[ggml_backend_buft_name(b)] = b;
        }
    }
    std::vector<std::string> patterns;
    patterns.reserve(cand.overrides.size());
    p.tensor_buft_overrides.clear();
    for (const auto & o : cand.overrides) {
        auto it = bufts.find(o.buft);
        if (it == bufts.end()) {
            LOG_ERR("%s: unknown buffer type %s\n", __func__, o.buft.c_str());
            return -1;
        }
        patterns.push_back(o.pattern);
        p.tensor_buft_overrides.push_back({ patterns.back().c_str(), it->second });
    }
    p.tensor_buft_overrides.push_back({ nullptr, nullptr });

    llama_model_params mparams = common_model_params_to_llama(p);
    mparams.no_alloc  = true;
    mparams.load_mode = LLAMA_LOAD_MODE_NONE;

    llama_model * model = llama_model_load_from_file(p.model.path.c_str(), mparams);
    if (model == nullptr) {
        LOG_ERR("%s: no_alloc load failed for candidate %s\n", __func__, cand.name.c_str());
        return -1;
    }

    verify_ud ud;
    ud.inv = &inv;
    ud.alloc = &alloc;
    ud.device_bufts = &device_bufts;
    for (size_t i = 0; i < inv.tensors.size(); i++) {
        ud.index_by_name[inv.tensors[i].name] = i;
    }

    llama_model_for_each_tensor(model, [](const ggml_tensor * t, void * vud) -> bool {
        verify_ud & u = *(verify_ud *) vud;
        const auto it = u.index_by_name.find(ggml_get_name(t));
        if (it == u.index_by_name.end() || it->second >= u.alloc->tensor_device.size()) {
            return true; // not in the inventory (e.g. a tensor the loader synthesized)
        }
        const int want = u.alloc->tensor_device[it->second];
        const std::string want_buft = want == fit_advisor_allocation::DEV_CPU ? "CPU" : u.device_bufts->at(want);
        std::string got = t->buffer ? ggml_backend_buft_name(ggml_backend_buffer_get_type(t->buffer)) : "(none)";
        // the CPU may pick a specialised buffer type such as a repacking one; those are host buffers
        const bool got_is_host = t->buffer && ggml_backend_buft_is_host(ggml_backend_buffer_get_type(t->buffer));
        const bool ok = got == want_buft || (want == fit_advisor_allocation::DEV_CPU && got_is_host);
        u.n_checked++;
        if (!ok) {
            u.n_bad++;
            if (u.n_bad <= 20) {
                LOG_WRN("%s: %s expected on %s, loader put it on %s\n", __func__, ggml_get_name(t), want_buft.c_str(), got.c_str());
            }
        }
        return true;
    }, &ud);

    llama_model_free(model);
    LOG_INF("%s: %s: %d tensors checked, %d misplaced\n", __func__, cand.name.c_str(), ud.n_checked, ud.n_bad);
    return ud.n_bad;
}
