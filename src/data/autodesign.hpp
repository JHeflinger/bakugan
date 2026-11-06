#pragma once
#include "data/design.hpp"
#include "data/job.hpp"
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace Bakugan {
    // Automatic design: fits superfrustum bodies to an input mesh in the open pose and, at the same time, hinges them so
    // the closed pose is as round as possible. One score covers both, and every change is judged by it, so the shapes,
    // the CSG and the hinges are tuned together rather than in stages.
    //
    // The fits are measured by silhouettes (by default): the input mesh's outline is drawn from several directions
    // once, and every design's open pose is drawn from the same directions and compared pixel by pixel (target pixels
    // covered, minus design pixels where the target has none); the closed pose is compared the same way against a disc
    // of the sphere's size around its centre. Each view must reach a minimum coverage, or the score suffers steeply.
    // The Volume option measures filled volumes and surfaces in 3D instead.
    //
    // The search is Stochastic Rewrite Descent (SRD; Kodnongbua, Zhang, Sharp, Schulz: "Design for Descent: What Makes
    // a Shape Grammar Easy to Optimize?", SIGGRAPH Asia 2025). It alternates two kinds of steps:
    //   descent   with the structure fixed, gradient steps on every continuous parameter (shape poses, sizes and
    //             profiles, pivots, axes, closing angles, blends)
    //   rewrites  sample a batch of grammar rewrites, give each a short lookahead, and apply every one that improves
    //             the score (best first, skipping any that conflict with one already taken), all at once
    // The grammar's rewrites come in reversible pairs:
    //   groups:   add / remove a body, split a shape off into a hinged body / merge a body into its parent,
    //             add / dissolve a blended feature
    //   shapes:   add / remove a superfrustum, move one to another body, remove or shrink one off the mesh
    //   material: make a shape (or a feature) add or cut material, carve a cut where two bodies overlap
    //   hinges:   add / remove a hinge, hang one from another body, turn its closing angle by a large step
    // New material and cuts are proposed where the last score found the mesh uncovered, the design sticking out of
    // it, the closed sphere dented or bulging, or two bodies overlapping. Collisions count in the open and the closed
    // pose only, not on the way between them. Hinge hardware (knuckles, leaves) is never part of the score.
    //
    // Unlike the paper, the score here is not differentiable code (it samples signed distance fields), so gradients
    // are estimated by SPSA: two scorings per step, whatever the number of parameters.
    // What the fit to the mesh and to the sphere is measured by.
    enum class AutoLoss {
        Silhouettes,                 // outlines from several directions: the open pose's against the mesh's, the
                                     // closed pose's against a disc of the sphere's size
        Volume                       // sampled volumes and surfaces in 3D
    };

    struct AutoParams {
        float sphereBias = 0.35f;    // 0: only the fit to the mesh matters, 1: only how round the closed shape is
        AutoLoss loss = AutoLoss::Silhouettes;
        int views = 6;               // Silhouettes: directions compared from, spread over a half sphere (the other
                                     // half would only see mirror images)
        int viewResolution = 64;     // Silhouettes: pixels across each view
        float minViewCoverage = 0.6f;    // Silhouettes: share of every view's target the design must cover; each
                                         // view short of it costs steeply, so none can be given up
        int iterations = 8000;       // scorings the search may spend (0 only scores the start design)
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
        int rewritesPerRound = 12;   // SRD: rewrites sampled and looked ahead per round (the paper uses 64)
        int stepsPerRound = 40;      // SRD: descent steps between rewrite rounds (the paper's code uses 25)
        int lookaheadSteps = 3;      // SRD: descent steps a rewrite gets before it is judged (the paper: 1 exact step)
        float stepSize = 1.0f;       // SRD: starting descent step, in parameter scale units
        float collisionWeight = 1.0f;    // how much bodies overlapping in the open or closed pose cost (0: ignored)
        int motionChecks = 4;        // in-between moments of the closing motion also checked for bodies overlapping
                                     // (and of the opening motion, when a hinge has its own opening timing); 0: only
                                     // the open and closed poses
        bool allowTiming = true;     // the grammar may change when each hinge turns (its share of the motion); only
                                     // matters with motion checks, since timing can't change the two end poses
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
        // silhouettes (when scored by them): per view, covered = target pixels the design covers / target pixels,
        // extra = design pixels on empty target pixels / target pixels; means over the views
        bool silhouettes = false;
        float viewCover = 0.0f;      // open pose against the mesh's outlines
        float viewExtra = 0.0f;
        float sphereViewCover = 0.0f;    // closed pose against the sphere's disc
        float sphereViewExtra = 0.0f;
        float worstView = 0.0f;      // the lowest coverage of any view that counts
        float coveragePenalty = 0.0f;    // what views short of the minimum coverage cost
        float offFrame = 0.0f;       // share of material outside the views' frame (unseen, so it counts as extra)
        float collision = 0.0f;      // bodies overlapping: open + closed + the mean of the moments between, each
                                     // half volume share, half depth
        float clashOpen = 0.0f;      // share of the open design's volume inside two bodies at once
        float clashClosed = 0.0f;    // the same, closed
        float clashMotion = 0.0f;    // the same, at the worst checked moment between them
        float clashMotionT = 0.0f;   // ... that moment (0 open .. 1 closed)
        bool clashMotionOpening = false; // ... and whether it is of the opening motion
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
        int iteration = 0;           // scorings so far
        int iterations = 0;          // ... allowed
        int accepted = 0;            // rewrites applied
        int rounds = 0;              // rewrite rounds
        float stepSize = 0.0f;       // descent step of the moment
        std::vector<float> history;  // best total, sampled over the run
        std::vector<unsigned char> views;    // Silhouettes: the best design's views as an RGBA image, open pose above
        int viewsWidth = 0;                  // the closed one; dark where the design covers the target, red where it
        int viewsHeight = 0;                 // misses it, blue where it covers nothing of it
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
