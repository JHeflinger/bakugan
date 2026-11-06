#pragma once
#include "data/design.hpp"
#include "data/job.hpp"
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace Bakugan {
    // Automatic design: fits superfrustum bodies to an input mesh in the open pose and, at the same time, hinges them so
    // the closed pose is as round as possible. One score covers both, and every change the search tries (the design
    // grammar below) is judged by it, so the shapes, the CSG and the hinges are tuned together rather than in stages:
    //   groups:   add / remove a body (a new hinged piece, a split, a merge into its parent), add / dissolve a feature
    //   shapes:   add / remove a superfrustum, move one to another body
    //   material: make a shape (or a feature) add or cut material, change a group's blend
    //   hinges:   add / remove a hinge, move its pivot, turn its axis, change its closing angle, hang it from another body
    //   shapes:   move, turn and resize a superfrustum, change its rounding, taper, bulge, dilation and shell
    // The search is simulated annealing over that grammar; new material and cuts are proposed where the last score
    // found the mesh uncovered, the design sticking out of it, the closed sphere dented or bulging, or two bodies
    // overlapping. Collisions count in the open and the closed pose only, not on the way between them. Hinge
    // hardware (knuckles, leaves) is never part of the score.
    struct AutoParams {
        float sphereBias = 0.35f;    // 0: only the fit to the mesh matters, 1: only how round the closed shape is
        int iterations = 8000;       // 0 only scores the start design
        int seedBodies = 4;          // starting from scratch: pieces the mesh is split into (k-means)
        int minBodies = 3;           // a design has to have this many hinged parts (else merging them all is the
                                     // easy way out of every collision, and nothing folds)
        int maxBodies = 8;
        int maxShapes = 18;
        bool fromCurrent = false;    // refine the design in the editor instead of starting from scratch
        bool allowCuts = true;
        bool allowFeatures = true;   // nested, blended groups
        float simplicity = 0.3f;     // 0..1, how much each shape and feature costs
        float sphereDiameter = 0.0f; // mm, 0: a sphere of the closed shape's own volume
        float collisionWeight = 1.0f;    // how much bodies overlapping in the open or closed pose cost (0: ignored)
        bool hingeHardware = false;  // give the result's hinges their knuckles and leaves (they are never scored)
        int quality = 1;             // 0 fast, 1 normal, 2 fine: how many sample points the score uses
        unsigned seed = 1;
    };

    // How good a design is. Lower total is better; the parts are 0 (perfect) .. 1.
    struct AutoScore {
        bool valid = false;
        float total = 1e9f;
        float base = 1e9f;           // total without collisions
        float fit = 1.0f;            // open pose against the mesh
        float sphere = 1.0f;         // closed pose against a sphere
        float penalty = 0.0f;        // complexity, unattached pivots, bodies on no hinge
        float fitIoU = 0.0f;         // volume overlap with the mesh (intersection over union, around the mesh), 0..1
        float fitCover = 0.0f;       // share of the mesh's volume the design fills
        float fitGap = 1.0f;         // mean distance of the mesh surface from the design surface (share of a cap)
        float fitOutside = 0.0f;     // share of the design's volume outside the mesh
        float fitExcess = 0.0f;      // design volume outside the mesh, relative to the mesh's volume
        float fitSpill = 0.0f;       // mean distance of the design's volume outside the mesh (share of 10% of its size)
        float sphereIoU = 0.0f;      // closed shape against the sphere, 0..1
        float sphereGap = 1.0f;      // mean distance of the sphere surface from the closed surface (share of a cap)
        float sphereSpill = 0.0f;    // mean distance of the closed material outside the sphere (share of its radius)
        float collision = 0.0f;      // bodies overlapping: open + closed, each half volume share, half depth
        float clashOpen = 0.0f;      // share of the open design's volume inside two bodies at once
        float clashClosed = 0.0f;    // the same, closed
        Vector3 centre = { 0, 0, 0 };   // the sphere, in the closed pose (world)
        float radius = 0.0f;
    };

    // The best design at one moment of a search.
    struct AutoFrame {
        Design design;
        AutoScore score;
        int iteration = 0;
    };

    // Shared with the UI while a search runs: the best design so far and how the score has gone.
    struct AutoLive {
        std::mutex mutex;
        int version = 0;             // bumped whenever `best` changes
        Design best;
        AutoScore score;
        int iteration = 0;
        int iterations = 0;
        int accepted = 0;
        float temperature = 0.0f;
        std::vector<float> history;  // best total, sampled over the run
        int recordFrames = 0;        // set before the run: keep the best design at this many evenly spaced moments
        std::vector<AutoFrame> frames;   // ... the first is the start, the last the result
    };

    struct AutoResult {
        bool ok = false;
        bool cancelled = false;      // stopped early: `design` is still the best found
        std::string message;
        Design design;
        AutoScore score;
    };

    // `mesh` is the input in design coordinates (millimetres). `start` is used when p.fromCurrent is set.
    AutoResult AutoDesign(const TriMesh& mesh, const Design& start, const AutoParams& p, JobControl& c,
                          std::shared_ptr<AutoLive> live = nullptr);
}
