#pragma once

// the advisor's own command-line options. they are parsed by the tool before the common parser sees the remaining
// arguments, so the shared option list stays what every example needs; the shared flags the advisor reads (-m, -c,
// -dev, -ot, -old, -fa, --spec-type, --fit-target, ...) are the server's, which is what makes an emitted plan exact

#include <string>
#include <vector>

struct fit_advisor_options {
    std::string workload = "chat";         // --optimize-for: workload preset
    std::string search_ubatch = "512,1024,2048"; // --search-ubatch: ubatch sizes the search may choose from
    bool        validate = false;          // --validate: load the chosen allocation for real, time it, search again with what it measured
    std::string validate_prompt;           // --validate-prompt: text file the validation prompt is built from
    bool        mtp = true;                // --no-mtp: never choose MTP drafting
    double      mtp_accept = 0.75;         // --mtp-accept: acceptance probability per drafted token up to the trained depth
    double      mtp_decay  = 0.85;         // --mtp-decay: factor on it beyond the trained depth, applied once
    double      expert_coverage = 0;       // --expert-coverage: share of a layer's experts a prompt ubatch touches, 0 = estimate
    bool        pin_cpu_weights = false;   // --pin-cpu-weights: CPU-resident weights in the devices' pinned host buffer type
    bool        remeasure  = false;        // --remeasure: discard cached device measurements
    bool        no_measure = false;        // --no-measure: skip the measurements, only project memory
    std::string emit_ini;                  // --emit-ini: write the chosen allocation as a preset section to this INI file
};

// consume the advisor's options from argv, leaving the rest for the common parser (argc and argv are rewritten in
// place). returns false with a message on a bad value; help is left to the common parser, which calls print_usage
bool fit_advisor_parse_options(int & argc, char ** argv, fit_advisor_options & opts, std::string & error);

// the advisor's options for --help, after the common parser's own
void fit_advisor_print_usage(int argc, char ** argv);
