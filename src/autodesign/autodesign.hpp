#pragma once
// Automatic design: superfrustum bodies on hinges, fitted so that the open pose looks like an input mesh and the
// closed pose looks like a sphere, compared by their silhouettes from several directions.
//
// The optimizer (autodesign.cpp) is Adam on every differentiable number of the design, plus a small grammar
// (grammar.hpp) that changes the design's structure now and then, in the manner of Stochastic Rewrite Descent.
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
        float total = 0.0f;              // the loss: (1 - bias) * fit + bias * sphere + coverage penalty
        float fit = 0.0f;                // open pose against the mesh: mean over views of (1 - covered) + extra
        float sphere = 0.0f;             // closed pose against the sphere, the same way
        float coveragePenalty = 0.0f;    // 10 per whole share a view is covered less than minViewCoverage
        float viewCover = 0.0f, viewExtra = 0.0f;              // means over the views: open vs mesh
        float sphereViewCover = 0.0f, sphereViewExtra = 0.0f;  // closed vs sphere
        float worstView = 0.0f;          // least covered view, of either pose
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
