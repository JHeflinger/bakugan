#pragma once
// Automatic design: superfrustum bodies on hinges, fitted so that the open pose looks like an input mesh and the
// closed pose looks like a sphere, compared by their silhouettes from several directions.
//
// The optimizer (autodesign.cpp) is Adam on every differentiable number of the design, plus a small grammar
// (grammar.hpp) that changes the design's structure now and then, in the manner of Stochastic Rewrite Descent. The
// loss (loss.hpp) is the silhouettes, plus bodies overlapping as the design opens and closes.
// README.md walks through it.
#include "data/design.hpp"
#include "data/job.hpp"
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace Bakugan {
    struct AutoParams {
        // what is wanted
        float sphereBias = 0.35f;        // 0: only the open pose matching the mesh counts .. 1: only the closed pose's roundness
        float sphereDiameter = 0.0f;     // mm; 0: a sphere of the mesh's volume
        int views = 6;                   // directions the silhouettes are compared from
        int viewResolution = 64;         // pixels across each view
        float minViewCoverage = 0.6f;    // every view's target must be covered at least this much (steep cost below)
        float collisionWeight = 1.0f;    // what bodies overlapping each other cost (0: ignored)
        int motionChecks = 8;            // poses checked for overlaps between open and closed (the two ends always are)
        float motionCheckBias = 0.5f;    // 0: those poses spread evenly over the times some hinge moves .. 1: placed by
                                         // how many hinges move at once
        // the optimizer
        int steps = 1500;                // gradient evaluations in all (the budget), rewrite lookaheads included
        float learningRate = 1.0f;       // Adam's step, in each parameter's own unit (see params.cpp)
        int stepsPerRound = 60;          // Adam steps between rewrite rounds
        int rewritesPerRound = 3;        // rewrites tried in a round
        int lookahead = 12;              // Adam steps each tried rewrite (and the unchanged design) gets before comparing
        int maxBodies = 8;
        bool fromCurrent = false;        // start from the editor's design instead of two hinged boxes
        unsigned seed = 1;
    };

    struct AutoScore {
        bool valid = false;
        float total = 0.0f;              // the loss: (1 - bias) * fit + bias * sphere + coverage penalty + collision weight * collision
        float fit = 0.0f;                // open pose against the mesh: mean over views of (1 - covered) + extra
        float sphere = 0.0f;             // closed pose against the sphere, the same way
        float coveragePenalty = 0.0f;    // 10 per whole share a view is covered less than minViewCoverage
        float viewCover = 0.0f, viewExtra = 0.0f;              // means over the views: open vs mesh
        float sphereViewCover = 0.0f, sphereViewExtra = 0.0f;  // closed vs sphere
        float worstView = 0.0f;          // least covered view, of either pose
        float collision = 0.0f;          // overlap depth: open + closed + the mean of the poses between (see loss.cpp)
        float clashOpen = 0.0f, clashClosed = 0.0f;    // volume inside two bodies at once, as a share of the sphere's
        float clashMotion = 0.0f;        // ... at the worst checked pose between
        float clashMotionT = 0.0f;       // ... which is this far through the closing motion
        std::vector<float> checkTimes;   // where in the motion (0 open .. 1 closed) the poses between were checked
        Vector3 centre = { 0, 0, 0 };    // the target sphere
        float radius = 0.0f;
    };

    struct AutoFrame {                   // a snapshot of the best design, for recordings
        Design design;
        AutoScore score;
        int iteration = 0;
    };

    // Shared with the UI while a run goes.
    struct AutoLive {
        std::mutex mutex;
        int version = 0;                 // bumped whenever `best` changes
        Design best;
        AutoScore score;
        int iteration = 0, iterations = 0;
        int rounds = 0, accepted = 0;    // rewrite rounds, and rewrites kept
        std::vector<float> history;      // loss over the run
        std::vector<unsigned char> views;    // RGBA picture of the best design's silhouettes against their targets
        int viewsWidth = 0, viewsHeight = 0;
        int recordFrames = 0;            // set before starting: this many evenly spaced snapshots go into `frames`
        std::vector<AutoFrame> frames;
    };

    struct AutoResult {
        bool ok = false;
        bool cancelled = false;
        Design design;
        AutoScore score;
        std::string message;
    };

    AutoResult AutoDesign(const TriMesh& mesh, const Design& start, const AutoParams& p, JobControl& c,
                          std::shared_ptr<AutoLive> live = nullptr);
}
