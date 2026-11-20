#ifndef FB_GRAMMAR_H
#define FB_GRAMMAR_H
// The rewrite rules. Each changes the design in place and returns true, or returns false when it can't apply.
//   split:   a body cut by a plane into two pieces (each one piece), a hinge on the new seam opened as far as the loss
//            likes best (the cut, where on the seam and the angle searched over a few dozen candidates)
//   merge:   a hinge that hardly turns goes, its child body becomes part of its parent
//   join:    a negligible body becomes part of its parent (the sphere stays whole); the hinges that hung from it hang
//            from its parent, each turned to where it took its piece as nearly as it can
//   carve:   a hollow (a capsule taken from the sphere) starts, at nothing, under the open pose's worst extra
//   uncarve: a hollow goes
#include "core/config.h"
#include "core/design.h"

typedef enum { RULE_SPLIT = 0, RULE_MERGE, RULE_JOIN, RULE_CARVE, RULE_UNCARVE, RULE_COUNT } Rule;
const char* rule_name(int rule);
bool rule_simplifies(int rule);
struct Evaluator;
bool rule_apply(int rule, Design* d, const Params* p, Rng* rng, struct Evaluator* ev, float collisionScale);
int rule_draw(const Params* p, Rng* rng);
#endif
