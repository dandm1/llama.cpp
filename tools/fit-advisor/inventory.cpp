#include "inventory.h"

#include "gguf.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <cinttypes>
#include <regex>
#include <stdexcept>

const char * fit_advisor_tensor_kind_name(fit_advisor_tensor_kind kind) {
    switch (kind) {
        case FIT_ADVISOR_TENSOR_ATTN:        return "attn";
        case FIT_ADVISOR_TENSOR_FFN:         return "ffn";
        case FIT_ADVISOR_TENSOR_FFN_EXPS:    return "ffn_exps";
        case FIT_ADVISOR_TENSOR_LAYER_OTHER: return "layer_other";
        case FIT_ADVISOR_TENSOR_TOKEN_EMBD:  return "token_embd";
        case FIT_ADVISOR_TENSOR_OUTPUT:      return "output";
        case FIT_ADVISOR_TENSOR_GLOBAL:      return "global";
    }
    return "?";
}

// read an integer metadata value of any integer width, or return def when the key is missing
static uint64_t get_kv_uint(const gguf_context * ctx, const std::string & key, uint64_t def) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) {
        return def;
    }
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8 (ctx, id);
        case GGUF_TYPE_INT8:   return gguf_get_val_i8 (ctx, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(ctx, id);
        case GGUF_TYPE_INT16:  return gguf_get_val_i16(ctx, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(ctx, id);
        case GGUF_TYPE_UINT64: return gguf_get_val_u64(ctx, id);
        case GGUF_TYPE_INT64:  return gguf_get_val_i64(ctx, id);
        default:
            throw std::runtime_error("metadata key '" + key + "' is not an integer");
    }
}

// like get_kv_uint but accepts an array of integers too, returning the maximum (some architectures store per-layer head counts)
static uint64_t get_kv_uint_or_arr_max(const gguf_context * ctx, const std::string & key, uint64_t def) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) {
        return def;
    }
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY) {
        return get_kv_uint(ctx, key, def);
    }
    const int64_t n = gguf_get_arr_n(ctx, id);
    const gguf_type at = gguf_get_arr_type(ctx, id);
    const void * data = gguf_get_arr_data(ctx, id);
    uint64_t ret = 0;
    for (int64_t i = 0; i < n; i++) {
        uint64_t v = 0;
        switch (at) {
            case GGUF_TYPE_UINT8:  v = ((const uint8_t  *) data)[i]; break;
            case GGUF_TYPE_INT8:   v = ((const int8_t   *) data)[i]; break;
            case GGUF_TYPE_UINT16: v = ((const uint16_t *) data)[i]; break;
            case GGUF_TYPE_INT16:  v = ((const int16_t  *) data)[i]; break;
            case GGUF_TYPE_UINT32: v = ((const uint32_t *) data)[i]; break;
            case GGUF_TYPE_INT32:  v = ((const int32_t  *) data)[i]; break;
            case GGUF_TYPE_UINT64: v = ((const uint64_t *) data)[i]; break;
            case GGUF_TYPE_INT64:  v = ((const int64_t  *) data)[i]; break;
            default: return def;
        }
        ret = std::max(ret, v);
    }
    return ret;
}

static std::string get_kv_str(const gguf_context * ctx, const std::string & key, const std::string & def) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) {
        return def;
    }
    return gguf_get_val_str(ctx, id);
}

// classify a tensor by its name, following the naming scheme in llama-arch.cpp
static fit_advisor_tensor classify(const std::string & name, ggml_type type, size_t nbytes) {
    static const std::regex re_layer(R"(^blk\.(\d+)\.(.+)$)");
    static const std::regex re_ffn(R"(^ffn_(gate|up|down|gate_up)(_shexp)?(\.weight|\.bias)$)");

    fit_advisor_tensor t = { name, type, nbytes, -1, FIT_ADVISOR_TENSOR_GLOBAL };

    if (name == "token_embd.weight") {
        t.kind = FIT_ADVISOR_TENSOR_TOKEN_EMBD;
        return t;
    }
    if (name == "output.weight") {
        t.kind = FIT_ADVISOR_TENSOR_OUTPUT;
        return t;
    }

    std::smatch m;
    if (!std::regex_match(name, m, re_layer)) {
        return t;
    }
    t.layer = std::stoi(m[1]);
    const std::string rest = m[2];

    if (rest.rfind("ffn_", 0) == 0 && rest.find("_exps") != std::string::npos) {
        t.kind = FIT_ADVISOR_TENSOR_FFN_EXPS;
    } else if (std::regex_match(rest, re_ffn)) {
        t.kind = FIT_ADVISOR_TENSOR_FFN;
    } else if (rest.rfind("attn_", 0) == 0 && rest.find("norm") == std::string::npos) {
        t.kind = FIT_ADVISOR_TENSOR_ATTN;
    } else {
        t.kind = FIT_ADVISOR_TENSOR_LAYER_OTHER;
    }
    return t;
}

static void add_shard(fit_advisor_inventory & inv, const std::string & path, bool first) {
    gguf_init_params gparams = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    gguf_context * ctx = gguf_init_from_file(path.c_str(), gparams);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to read GGUF metadata from " + path);
    }

    if (first) {
        inv.arch = get_kv_str(ctx, "general.architecture", "");
        if (inv.arch.empty()) {
            gguf_free(ctx);
            throw std::runtime_error("GGUF has no general.architecture key: " + path);
        }
        inv.n_layer       = (uint32_t) get_kv_uint(ctx, inv.arch + ".block_count",          0);
        inv.n_layer_nextn = (uint32_t) get_kv_uint(ctx, inv.arch + ".nextn_predict_layers", 0);
        inv.n_expert      = (uint32_t) get_kv_uint(ctx, inv.arch + ".expert_count",         0);
        inv.n_expert_used = (uint32_t) get_kv_uint(ctx, inv.arch + ".expert_used_count",    0);
        inv.n_embd        = (uint32_t) get_kv_uint(ctx, inv.arch + ".embedding_length",     0);
        inv.n_head        = (uint32_t) get_kv_uint_or_arr_max(ctx, inv.arch + ".attention.head_count",    0);
        inv.n_head_kv     = (uint32_t) get_kv_uint_or_arr_max(ctx, inv.arch + ".attention.head_count_kv", inv.n_head);
        inv.head_size     = (uint32_t) get_kv_uint(ctx, inv.arch + ".attention.key_length",
                                                   inv.n_head > 0 ? inv.n_embd / inv.n_head : 0);
        inv.n_ctx_train   = (uint32_t) get_kv_uint(ctx, inv.arch + ".context_length",       0);
        inv.n_split       = (uint32_t) get_kv_uint(ctx, "split.count",                      1);
        if (inv.n_split == 0) {
            inv.n_split = 1;
        }
    }

    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    for (int64_t i = 0; i < n_tensors; i++) {
        inv.tensors.push_back(classify(gguf_get_tensor_name(ctx, i), gguf_get_tensor_type(ctx, i), gguf_get_tensor_size(ctx, i)));
    }

    gguf_free(ctx);
}

fit_advisor_inventory fit_advisor_inventory_load(const std::string & path) {
    fit_advisor_inventory inv;
    inv.path = path;

    add_shard(inv, path, true);

    if (inv.n_split > 1) {
        // the given file may be any shard, derive the prefix and read the others
        gguf_init_params gparams = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
        gguf_context * ctx = gguf_init_from_file(path.c_str(), gparams);
        const int split_no = (int) get_kv_uint(ctx, "split.no", 0);
        gguf_free(ctx);

        char prefix[4096];
        if (llama_split_prefix(prefix, sizeof(prefix), path.c_str(), split_no, (int) inv.n_split) == 0) {
            throw std::runtime_error("split.count > 1 but the file name is not a split path: " + path);
        }
        for (int i = 0; i < (int) inv.n_split; i++) {
            if (i == split_no) {
                continue;
            }
            char shard[4096];
            llama_split_path(shard, sizeof(shard), prefix, i, (int) inv.n_split);
            add_shard(inv, shard, false);
        }
    }

    // group into layers; MTP layers, if any, sit after the regular ones
    int32_t il_max = -1;
    for (const auto & t : inv.tensors) {
        il_max = std::max(il_max, t.layer);
    }
    const uint32_t n_layers_seen = (uint32_t) (il_max + 1);
    if (n_layers_seen > inv.n_layer + inv.n_layer_nextn) {
        // metadata did not declare the extra blocks, trust the tensors
        inv.n_layer_nextn = n_layers_seen - inv.n_layer;
    }
    inv.layers.assign(inv.n_layer + inv.n_layer_nextn, {});

    for (const auto & t : inv.tensors) {
        inv.total += t.nbytes;
        inv.bytes_by_type[t.type] += t.nbytes;
        switch (t.kind) {
            case FIT_ADVISOR_TENSOR_ATTN:        inv.layers[t.layer].attn     += t.nbytes; break;
            case FIT_ADVISOR_TENSOR_FFN:         inv.layers[t.layer].ffn      += t.nbytes; break;
            case FIT_ADVISOR_TENSOR_FFN_EXPS:    inv.layers[t.layer].ffn_exps += t.nbytes; break;
            case FIT_ADVISOR_TENSOR_LAYER_OTHER: inv.layers[t.layer].other    += t.nbytes; break;
            case FIT_ADVISOR_TENSOR_TOKEN_EMBD:  inv.token_embd               += t.nbytes; break;
            case FIT_ADVISOR_TENSOR_OUTPUT:      inv.output                   += t.nbytes; break;
            case FIT_ADVISOR_TENSOR_GLOBAL:      inv.global                   += t.nbytes; break;
        }
    }

    return inv;
}

size_t fit_advisor_inventory::layer_bytes(uint32_t il_begin, uint32_t il_end, fit_advisor_tensor_kind kind) const {
    size_t ret = 0;
    for (uint32_t il = il_begin; il < il_end && il < layers.size(); il++) {
        switch (kind) {
            case FIT_ADVISOR_TENSOR_ATTN:        ret += layers[il].attn;     break;
            case FIT_ADVISOR_TENSOR_FFN:         ret += layers[il].ffn;      break;
            case FIT_ADVISOR_TENSOR_FFN_EXPS:    ret += layers[il].ffn_exps; break;
            case FIT_ADVISOR_TENSOR_LAYER_OTHER: ret += layers[il].other;    break;
            default: break;
        }
    }
    return ret;
}

std::vector<ggml_type> fit_advisor_inventory::weight_types(double min_share) const {
    std::vector<std::pair<ggml_type, size_t>> v(bytes_by_type.begin(), bytes_by_type.end());
    std::sort(v.begin(), v.end(), [](const auto & a, const auto & b) { return a.second > b.second; });
    std::vector<ggml_type> ret;
    for (const auto & [type, bytes] : v) {
        if (total > 0 && (double) bytes / total >= min_share) {
            ret.push_back(type);
        }
    }
    return ret;
}

void fit_advisor_inventory_print(const fit_advisor_inventory & inv) {
    constexpr double MiB = 1024.0 * 1024.0;

    LOG_INF("%s: model %s\n", __func__, inv.path.c_str());
    LOG_INF("%s: arch = %s, n_layer = %" PRIu32 ", n_layer_nextn = %" PRIu32 ", n_expert = %" PRIu32 ", n_ctx_train = %" PRIu32 ", shards = %" PRIu32 "\n",
        __func__, inv.arch.c_str(), inv.n_layer, inv.n_layer_nextn, inv.n_expert, inv.n_ctx_train, inv.n_split);
    LOG_INF("%s: %zu tensors, %.1f MiB total: token_embd %.1f, output %.1f, global %.1f\n",
        __func__, inv.tensors.size(), inv.total / MiB, inv.token_embd / MiB, inv.output / MiB, inv.global / MiB);
    LOG_INF("%s: n_embd = %" PRIu32 ", n_head = %" PRIu32 ", n_head_kv = %" PRIu32 ", head_size = %" PRIu32 ", experts used/total = %" PRIu32 "/%" PRIu32 "\n",
        __func__, inv.n_embd, inv.n_head, inv.n_head_kv, inv.head_size, inv.n_expert_used, inv.n_expert);
    {
        std::string mix;
        for (const ggml_type type : inv.weight_types(0.0)) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%s%s %.1f%%", mix.empty() ? "" : ", ", ggml_type_name(type), 100.0 * inv.bytes_by_type.at(type) / inv.total);
            mix += buf;
        }
        LOG_INF("%s: types by bytes: %s\n", __func__, mix.c_str());
    }

    const uint32_t n_all = (uint32_t) inv.layers.size();
    LOG_INF("%s: per layer [MiB]: attn %.1f, ffn %.1f, ffn_exps %.1f, other %.1f (sums over %" PRIu32 " layers)\n",
        __func__,
        inv.layer_bytes(0, n_all, FIT_ADVISOR_TENSOR_ATTN)        / MiB,
        inv.layer_bytes(0, n_all, FIT_ADVISOR_TENSOR_FFN)         / MiB,
        inv.layer_bytes(0, n_all, FIT_ADVISOR_TENSOR_FFN_EXPS)    / MiB,
        inv.layer_bytes(0, n_all, FIT_ADVISOR_TENSOR_LAYER_OTHER) / MiB,
        n_all);

    // layers are usually uniform, only list the ones that differ from the first
    for (uint32_t il = 1; il < n_all; il++) {
        const auto & a = inv.layers[il];
        const auto & b = inv.layers[0];
        if (a.attn != b.attn || a.ffn != b.ffn || a.ffn_exps != b.ffn_exps || a.other != b.other) {
            LOG_INF("%s: layer %" PRIu32 " differs from layer 0: attn %.1f, ffn %.1f, ffn_exps %.1f, other %.1f MiB%s\n",
                __func__, il, a.attn / MiB, a.ffn / MiB, a.ffn_exps / MiB, a.other / MiB, il >= inv.n_layer ? " (MTP)" : "");
        }
    }
}
