#pragma once
// The grammar: changes to the design's structure that Adam can't make (it only moves numbers).
//
// A rule is a function that changes a design, or returns false when it can't apply. The optimizer draws rules by
// weight from the table in grammar.cpp; to add one, write its function and add a row.
#include "loss.hpp"
#include <random>

namespace Bakugan {
    struct RuleContext {
        const Loss& loss;            // the targets (points on the mesh, its size)
        const AutoParams& p;
        std::mt19937& rng;
    };

    struct Rule {
        const char* name;
        float weight;                // how often it is drawn, against the other rules
        bool (*apply)(Design& d, const RuleContext& c);
    };

    const std::vector<Rule>& Rules();
    // Draws a rule by weight and applies it; the rule's name, or nullptr when it didn't apply.
    const char* ApplyRandomRule(Design& d, const RuleContext& c);
}
