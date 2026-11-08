#pragma once
// The grammar: changes to the design's structure that Adam can't make (it only moves numbers).
//
// A rule is a function that changes a design, or returns false when it can't apply. The optimizer draws rules by
// weight from the table in grammar.cpp; to add one, write its function and add a row. Rules that add something have
// to improve the loss to be kept; rules that take something away (simplifies) only have to not make it worse.
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
        bool simplifies = false;     // it takes something away: kept when the loss is no worse than rewriteMargin above
                                     // leaving the design alone (rules that add must beat it by that much)
    };

    const std::vector<Rule>& Rules();
    // Draws a rule by weight (weights in Rules() order; missing ones: the rule's default) and applies it; the rule, or
    // nullptr when it didn't apply.
    const Rule* ApplyRandomRule(Design& d, const RuleContext& c, const std::vector<float>& weights = {});
}
