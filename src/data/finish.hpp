#pragma once
// Finishing passes: what makes a carved sphere something that can be printed and played with. Each takes a design and
// gives back a changed one (ordinary shapes and groups: it saves, shows, checks and exports as any design does).
//
//   CarveCollisions   wherever two bodies still run into each other on the way between open and closed, the overlap
//                     is cut out of one of them (the one it shows less on, closed).
//   AddHinges         the hinges themselves: each axis gets the smallest barrel that can be printed (three knuckles on
//                     a printed-in-place pin), put where both of its bodies are along the axis and, for a carved
//                     sphere, far enough under the surface to be inside it.
//   AddLedges         every hinge but the first to open gets a small ledge of a piece that moves away earlier, lying
//                     over the piece the hinge turns: it can't open until that piece has. So one hinge holds the whole
//                     design shut, and letting it go lets the rest unfurl in turn.
//   AddLatch          a press latch on that one hinge: a small rocker printed into the surface whose catch lies over
//                     the lid's edge (or, in the lid, hooks under the other body's), and steps back off it when the
//                     rocker's other end is pressed in.
//
// Ledges and the latch are cut from the sphere itself (their material changes hands between bodies, nothing is added),
// so the closed sphere's surface stays flush. They need to know the sphere: carved designs carry it (FindSphere).
#include "data/design.hpp"
#include "data/job.hpp"
#include <string>
#include <vector>

namespace Bakugan {
    struct SphereInfo {
        bool ok = false;
        Vector3 centre = { 0, 0, 0 };
        float radius = 0.0f;
    };
    // The sphere a carved design's pieces are cut from (its bodies' "Sphere" shapes).
    SphereInfo FindSphere(const Design& d);

    struct CarveParams {
        int checksPerHinge = 32;         // poses looked at in every hinge's turn (and the open pose)
        float clearance = 0.3f;          // mm left around what is carved away
        int rounds = 4;                  // carve, check again (between the last poses), carve what is left, ...
    };

    struct LedgeParams {
        float size = 0.0f;               // mm a ledge reaches over the piece it holds (0: 12% of the sphere's radius);
                                         // it is 2.5 times as wide, 0.6 times as thick, and as long again in its own piece
        float reach = 0.0f;              // mm of another piece's surface a ledge may cross to get there (0: 35% of the radius)
    };

    struct HingeParams {
        int hinge = -1;                  // hinge id (-1: every hinge)
        float scale = 1.0f;              // its size, against the smallest that prints (a 1.6 mm pin in a 3.8 mm barrel)
        float clearance = 0.3f;          // mm between the pin and the knuckles around it, and between knuckles
        float reach = 0.0f;              // mm a body may be from the axis and still be joined to its knuckles, by a
                                         // leaf that long (0: 1.2 mm at the default size, just enough to tie them in)
        bool flush = true;               // a carved sphere: the axis goes far enough under the surface for the barrel
                                         // to be inside the sphere (it moves, so the motion changes a little near it)
        PinStyle pin = PinStyle::Fused;  // printed in place; or a separate pin (flush: with a channel to push it in by)
    };

    // The hardware AddHinges builds, for a sphere of this radius (0: not a carved sphere).
    struct HingeSizes {
        float pinRadius = 0.0f, radius = 0.0f, clearance = 0.0f;
        int knuckles = 3;
        float length = 0.0f;             // along the axis
        float leafLength = 0.0f, leafThickness = 0.0f;
        float pivotRadius = 0.0f;        // flush: the farthest from the sphere's centre the middle of the axis can be
        float relief = 0.0f;             // flush: within this of the axis, its two bodies are in each other's way as
                                         // they turn (the axis being under the seam); AddHinges cuts that room
    };
    HingeSizes PlanHinge(float sphereRadius, const HingeParams& p);

    struct LatchParams {
        int hinge = -1;                  // hinge id (-1: the first to open)
        float scale = 1.0f;              // its size, against the default for the sphere
        float clearance = 0.35f;         // mm between the rocker and what is around it
    };

    struct FinishResult {
        bool ok = false;                 // the design was changed
        bool cancelled = false;
        Design design;                   // the changed design (the given one when !ok)
        std::string message;
        // CarveCollisions
        int cuts = 0;                    // shapes added
        double before = 0.0, after = 0.0;    // mm^3 inside two bodies at once, at the worst pose
        // AddLedges
        int dependents = 0;              // hinges that should be held (every moving one but the first to open)
        int held = 0;                    // ... of them, those that now are
        int root = -1;                   // the hinge that opens first (id)
        std::vector<std::string> unheld; // the names of the rest
        // AddHinges
        int built = 0;                   // hinges that got hardware
        int relieved = 0;                // cuts that make room next to them
        float moved = 0.0f;              // the farthest an axis was moved (mm)
    };

    FinishResult CarveCollisions(const Design& d, const CarveParams& p, JobControl* c = nullptr);
    FinishResult AddHinges(const Design& d, const HingeParams& p, JobControl* c = nullptr);
    FinishResult AddLedges(const Design& d, const LedgeParams& p, JobControl* c = nullptr);
    FinishResult AddLatch(const Design& d, const LatchParams& p, JobControl* c = nullptr);

    // Whether a hinge is held shut: with everything closed, starting to open it runs into something (more than
    // `tolerance` mm^3 of overlap by the time it has turned a few degrees).
    bool HeldShut(const Design& d, int hingeIndex, float tolerance = 0.5f, double* overlap = nullptr);
}
