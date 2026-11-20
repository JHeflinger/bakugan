#ifndef FB_ADAM_H
#define FB_ADAM_H
#include "eval.h"
// One Adam step over every parameter of the design, the moments kept in the design itself (so a copy of the design
// carries its optimizer state along, and a new plane or hinge starts with none).
void adam_step(Design* d, const Grad* g, const Params* p, float lrScale);
#endif
