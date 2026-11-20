#ifndef FB_FEATURES_H
#define FB_FEATURES_H
// Feature restoration (post processing, see features.c): the input's surface put back onto the parts it opens into,
// where it doesn't collide; flattened and sunk (feature_flatten, feature_sink) where it would. Kept per body as an offset
// grid (Body.detail). Returns how many parts got features; msg says what was done.
#include "core/config.h"
#include "core/design.h"

int restore_features(Design* d, const Params* p, char* msg, int msgSize);
#endif
