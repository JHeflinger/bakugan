#ifndef FB_POST_H
#define FB_POST_H
// Post processing a design for printing, in this order:
//   1. fragments: any part of a body apart from the rest of it goes
//   1b. features (optional): the input's surface restored onto the parts, where it doesn't collide (features.h)
//   2. hinges: knuckles alternating between the two bodies along each axis, a pin fused into the parent's, a bore
//      in the child's, and room cleared around each body's knuckles in the other
//   3. carving: whatever still collides along the motion (checked densely) is carved from one of the two bodies,
//      smoothly, with a clearance (a swept volume, kept as a grid per body)
//   4. ledges: every hinge but the root's is held shut by a body that opens before it: where none does yet, a
//      ledge on such a body reaches over the hinge's piece (into a recess in it), so it can't open until that body
//      has moved away
//   5. latch: on the root hinge, a press latch flush with the sphere: a flexible tab in the opening piece whose hook
//      sits under a lip of the other; pressing the tab frees the hook
//   6. fragments again (carving can leave bits)
#include "core/config.h"
#include "core/design.h"

typedef struct {
    int fragmentsRemoved;
    int hingesBuilt;
    int carvedPairs;
    int ledges, ledgesFailed, alreadyHeld;
    bool latched;
    float overlapBefore, overlapAfter;   // worst overlap along the motion, share of the volume
    int featureParts;                    // parts the input's features were restored on
    char featureText[512];
    char text[2048];
} PostReport;

bool post_process(Design* d, const Params* p, PostReport* report);
// The worst overlap along the motion (share of the volume), checked at `checks` poses per hinge turn.
float post_overlap(const Design* d, int checks);
#endif

#ifndef FB_EXPORT_DECL
#define FB_EXPORT_DECL
// Every body as an STL (mm), in the open or closed pose: <dir>/<stem>_<pose>_part<k>.stl. Returns how many were
// written; msg says where.
int export_stl(const Design* d, const Params* p, const char* dir, const char* stem, char* msg, int msgSize);
#endif
