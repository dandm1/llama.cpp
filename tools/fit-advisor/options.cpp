#include "options.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

struct option_spec {
    const char * name;
    const char * value; // nullptr for a switch
    const char * help;
};

const option_spec specs[] = {
    { "--optimize-for",    "PRESET", "workload preset to optimise for: chat, rag, batch, agent (default: chat)" },
    { "--search-ubatch",   "N,...",  "ubatch sizes the search may choose from, a single value pins it (default: 512,1024,2048)" },
    { "--validate",        nullptr,  "load the chosen allocation for real, time its steps and measure its memory, then search again with what was measured" },
    { "--validate-prompt", "FILE",   "text the validation prompt is built from (default: a built-in paragraph)" },
    { "--no-mtp",          nullptr,  "never choose MTP drafting (default: the search decides per allocation when the model has MTP layers)" },
    { "--mtp-accept",      "P",      "assumed acceptance probability per drafted MTP token up to the model's trained MTP depth (default: 0.75)" },
    { "--mtp-decay",       "D",      "factor on the acceptance beyond the trained MTP depth, applied once (default: 0.85)" },
    { "--expert-coverage", "F",      "share of a layer's experts a prompt ubatch routes to, (0, 1]; default: estimated, measured by --validate" },
    { "--pin-cpu-weights", nullptr,  "place CPU-resident weights in the devices' pinned host buffer type" },
    { "--remeasure",       nullptr,  "discard the cached device measurements and measure again" },
    { "--no-measure",      nullptr,  "skip the device measurements, only project memory" },
    { "--emit-ini",        "PATH",   "write the chosen allocation as a preset section to this INI file (created or updated in place)" },
};

double parse_unit(const char * flag, const char * v, double lo, double hi, bool lo_open, std::string & error) {
    char * end = nullptr;
    const double x = std::strtod(v, &end);
    if (end == v || *end != '\0' || x < lo || x > hi || (lo_open && x == lo)) {
        error = std::string(flag) + " must be a number in " + (lo_open ? "(" : "[") + std::to_string(lo) + ", " + std::to_string(hi) + "]";
        return 0;
    }
    return x;
}

} // namespace

bool fit_advisor_parse_options(int & argc, char ** argv, fit_advisor_options & opts, std::string & error) {
    int out = 1;
    for (int i = 1; i < argc; i++) {
        const char * a = argv[i];
        const option_spec * spec = nullptr;
        for (const auto & s : specs) {
            if (std::strcmp(a, s.name) == 0) {
                spec = &s;
                break;
            }
        }
        if (!spec) {
            argv[out++] = argv[i];
            continue;
        }
        const char * v = nullptr;
        if (spec->value) {
            if (i + 1 >= argc) {
                error = std::string(a) + " needs a value";
                return false;
            }
            v = argv[++i];
        }
        const std::string name = a;
        if      (name == "--optimize-for")    opts.workload = v;
        else if (name == "--search-ubatch")   opts.search_ubatch = v;
        else if (name == "--validate")        opts.validate = true;
        else if (name == "--validate-prompt") opts.validate_prompt = v;
        else if (name == "--no-mtp")          opts.mtp = false;
        else if (name == "--mtp-accept")      opts.mtp_accept = parse_unit(a, v, 0.0, 1.0, false, error);
        else if (name == "--mtp-decay")       opts.mtp_decay  = parse_unit(a, v, 0.0, 1.0, true,  error);
        else if (name == "--expert-coverage") opts.expert_coverage = parse_unit(a, v, 0.0, 1.0, true, error);
        else if (name == "--pin-cpu-weights") opts.pin_cpu_weights = true;
        else if (name == "--remeasure")       opts.remeasure = true;
        else if (name == "--no-measure")      opts.no_measure = true;
        else if (name == "--emit-ini")        opts.emit_ini = v;
        if (!error.empty()) {
            return false;
        }
    }
    argc = out;
    argv[argc] = nullptr;
    return true;
}

void fit_advisor_print_usage(int argc, char ** argv) {
    (void) argc; (void) argv;
    printf("\nfit-advisor options:\n");
    for (const auto & s : specs) {
        char head[64];
        snprintf(head, sizeof(head), "%s %s", s.name, s.value ? s.value : "");
        printf("  %-34s %s\n", head, s.help);
    }
    printf("\n");
}
