#pragma once
// Design edits and queries the grammar is built from. They know nothing about the search: each takes a design and
// changes or reads it, keeping its structure valid (the hinge tree whole, every body with material).
#include "common.hpp"

namespace Bakugan::Auto {
    std::vector<int> Roots(const Design& d);                          // group ids of the bodies on no hinge
    int MaterialShapes(const Design& d, int bodyId);                  // shapes adding material to the body
    int NearestBody(const Design& d, Vector3 p, int except = -1);     // group index, in the design pose
    Vector3 BodyHint(const Design& d, int bodyId);                    // a point inside the body (its biggest shape)
    // Where two bodies meet between a (in A) and b (in B): the point on the segment as far from both.
    Vector3 PivotBetween(const Design& d, int bodyA, int bodyB, Vector3 a, Vector3 b);

    // Takes a body out, keeping the hinge tree whole: its children hang from its parent instead (or, for a root, from
    // its first child, which becomes the root).
    void RemoveBody(Design& d, int bodyId);
    // Folds a hinged body back into its parent: its shapes and features join the parent, its children hang there.
    bool MergeBody(Design& d, int bodyId);

    // Cuts a shape in two across local axis `axis` (0 x, 1 y, 2 z), `t` of the way along it: *a is the part towards -axis,
    // *b the rest. Across z the taper carries on through both halves, so together they are exactly the shape. Across x
    // or y they are exact for a sharp, untapered shape; rounding leaves small notches at the cut and a taper leans each
    // half towards its own middle (1-3% of the volume around the shape, for the descent to tune). A bulged shape is
    // never split across z (its bend runs along z): across its longer side instead. False when the halves would be
    // degenerate.
    bool SplitShape(const SuperFrustum& f, int axis, float t, SuperFrustum* a, SuperFrustum* b);

    // Puts a pivot, an axis and a closing turn on hinge `index`. A fresh hinge also starts open at 0 and gets hardware
    // sized to the model (switched off: the score ignores hardware, DesignSpace::Finalize decides).
    void SetupHinge(Design& d, int index, Vector3 pivot, Vector3 axis, float angle, bool fresh, float L);
    // A hinge's timing windows kept inside the motion, each a tenth of it at least (a hinge that snaps over in an
    // instant could swing through another body between two checked moments).
    void ClampTiming(Hinge& h);
}
