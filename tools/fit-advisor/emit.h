#pragma once

// write a chosen allocation as an INI preset section that llama-server (router mode, --models-preset) and the
// system-level config.ini understand; built through the shared preset writer so the keys are always ones the parser accepts

#include "probe.h"

#include <cstdint>
#include <string>

struct fit_advisor_emit_result {
    bool        ok = false;
    std::string error;
    std::string path;
    std::string section;
    std::string ini;      // the section text that was written
};

// model_path: what the section's model key points at; n_batch_base: the -b to write when the candidate sets a ubatch
// the file is created if missing; an existing section with the same name is replaced, everything else is kept
// after writing, the file is reloaded and every written option compared, so a silent truncation cannot pass
fit_advisor_emit_result fit_advisor_emit_ini(const std::string & path, const std::string & section, const std::string & model_path,
                                             const fit_advisor_candidate & cand, int32_t n_batch_base);
