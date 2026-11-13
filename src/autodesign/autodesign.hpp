#pragma once
// Automatic design: a sphere carved into pieces and hinged (carving.hpp), so that it opens into the shape of an input
// mesh. Closed, the pieces are the sphere again, always; open, they're compared with the mesh (which may be moved and
// scaled to suit) by their silhouettes from several directions, or by DINOv3.
//
// The optimizer (autodesign.cpp) is Adam on every differentiable number of the carving, plus a small grammar
// (grammar.hpp) that adds and removes cuts now and then, in the manner of Stochastic Rewrite Descent. The loss
// (loss.hpp) is the open pose's look, plus pieces overlapping as the design opens and closes, plus pivots off their
// pieces. README.md walks through it.
#include "carving.hpp"
#include "data/design.hpp"
#include "data/job.hpp"
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace Bakugan {
    // How the semantic loss draws its views (the design's and the mesh's alike), grey on dark:
    enum class SemanticLook {
        Silhouette,                      // white where covered
        Depth,                           // nearer is brighter
        Shaded,                          // a grey material under a key and a fill light (from the viewer's upper left
                                         // and right), on a dark grey background
    };

    struct AutoParams {
        // what is wanted
        float sphereDiameter = 0.0f;     // mm; 0: a sphere of the mesh's volume (it sits at the mesh's centre of volume)
        bool moveTarget = true;          // the mesh it's compared with may be moved, to wherever suits the design best
        bool scaleTarget = true;         // the mesh is scaled (about its centre of volume) to the sphere's volume, so the
                                         // sphere's material fills it exactly; off, it's compared at its own size. The scale
                                         // isn't optimized: a free one let the optimizer shrink the mesh (to ~0.6x) so the
                                         // sphere covered it easily, with ~4x the material it should have
        int views = 6;                   // directions the silhouettes are compared from
        int viewResolution = 64;         // pixels across each view
        float minViewCoverage = 0.6f;    // every view's target must be covered at least this much (steep cost below)
        float collisionWeight = 1.0f;    // what pieces overlapping each other cost (0: ignored)
        float hingeWeight = 2.0f;        // what a joint's pivot being away from its two pieces costs (0: ignored)
        float surfaceWeight = 1.0f;      // what hollows carving into the sphere's surface cost (0: ignored)
        int motionChecks = 8;            // poses checked for overlaps between open and closed (open always is; closed
                                         // needs no check: the pieces fill the sphere side by side)
        float motionCheckBias = 0.5f;    // 0: those poses spread evenly over the times some hinge moves .. 1: placed by
                                         // how many hinges move at once
        float silhouetteWeight = 1.0f;   // the silhouette comparison's share of the loss (0: only the semantic one)
        // in 3D (shape.cpp, transport.cpp): the open design against the placed mesh
        float chamferWeight = 0.0f;      // the Chamfer distance between their surfaces, both ways (0: off)
        float sdfWeight = 0.0f;          // their signed distances agreeing at points in and around the mesh (0: off)
        int shapeSamples = 3000;         // the mesh's points for those two (a third on its surface)
        float transportWeight = 0.0f;    // optimal transport: how far the sphere's material travels to fill the mesh (0: off)
        int transportPoints = 1024;      // points of material, and points filling the mesh, for it
        // the semantic loss: the views drawn and compared by DINOv3's embeddings (see dino.hpp, semantic.cpp)
        float semanticWeight = 0.0f;     // its share of the loss (0: off)
        std::string semanticModel;       // the .onnx tools/autodesign/export_dinov3.py writes (it sets the views' size)
        std::string onnxRuntime;         // the ONNX Runtime library (empty: the platform's default name, found as usual)
        bool semanticGpu = false;        // run DINOv3 with CUDA, if this ONNX Runtime has it
        SemanticLook semanticLook = SemanticLook::Shaded;
        float patchWeight = 0.0f;        // 0: compare the global (CLS) embeddings .. 1: the patch embeddings, place by place
        int semanticSamples = 4000;      // shaded pixels sampled for the gradient each step (their rest is the edges)
        // the optimizer
        int steps = 1500;                // gradient evaluations in all (the budget), rewrite lookaheads included
        float learningRate = 1.0f;       // Adam's step, in each parameter's own unit (see carving.cpp)
        int stepsPerRound = 60;          // Adam steps between rewrite rounds
        int rewritesPerRound = 3;        // rewrites tried in a round
        int lookahead = 12;              // Adam steps each tried rewrite (and the unchanged design) gets before comparing
        float rewriteMargin = 0.005f;    // a rewrite is kept only if its loss is this much (a share) below the unchanged design's
        int maxPieces = 8;
        int maxHollows = 8;
        std::vector<float> ruleWeights;  // how often each grammar rule is drawn, in AutoRules() order (empty: their defaults)
        unsigned seed = 1;
    };

    struct AutoScore {
        bool valid = false;
        float total = 0.0f;              // the loss: silhouette weight * (fit + coverage penalty) + semantic weight * semantic
                                         //           + collision weight * collision + hinge weight * hinges
                                         //           + surface weight * surface + chamfer, sdf, transport weights * theirs
        float fit = 0.0f;                // open pose against the placed mesh: mean over views of (1 - covered) + extra
        float coveragePenalty = 0.0f;    // 10 per whole share a view is covered less than minViewCoverage
        float viewCover = 0.0f, viewExtra = 0.0f;              // means over the views
        float worstView = 0.0f;          // least covered view
        float collision = 0.0f;          // overlap depth: open + the mean of the poses between (see loss.cpp)
        float clashOpen = 0.0f;          // volume inside two pieces at once, open, as a share of the sphere's
        float clashMotion = 0.0f;        // ... at the worst checked pose between
        float clashMotionT = 0.0f;       // ... which is this far through the closing motion
        std::vector<float> checkTimes;   // where in the motion (0 open .. 1 closed) the poses between were checked
        float hinges = 0.0f;             // pivots away from their pieces: the sum of how far, beyond 1% of the mesh's
                                         // size, each pivot is outside each of its two pieces, over the mesh's size
        float hingeGap = 0.0f;           // the farthest a pivot is outside one of its pieces (mm)
        float surface = 0.0f;            // hollows into the sphere's surface: over points just under it, the mean of
                                         // d / (d + 1% of the mesh's size), d how deep inside a hollow it is
        float surfaceCarved = 0.0f;      // the share of those points inside a hollow
        float chamfer = 0.0f;            // mesh -> design + design -> mesh, each a mean distance over 0.1 of the mesh's size
        float sdf = 0.0f;                // the signed distances' disagreement, likewise
        float meshGap = 0.0f;            // mean distance from the mesh's surface to the design's (mm)
        float designGap = 0.0f;          // ... from the design's surface to the mesh's (mm)
        float transport = 0.0f;          // optimal transport's cost (squared distances, over (0.1 L)^2)
        float transportMove = 0.0f;      // how far the material travels, root mean square (mm)
        float transportUnfilled = 0.0f;  // the share of the placed mesh's volume the material doesn't reach
        float semantic = 0.0f;           // DINOv3: mean over the views of the open pose's loss against the placed mesh
        float worstSemantic = 0.0f;      // the worst single view's
        Placement target;                // where the mesh was placed when this was scored
        Vector3 centre = { 0, 0, 0 };    // the sphere's (the closed pose)
        float radius = 0.0f;
    };

    struct AutoFrame {                   // a snapshot of the best design, for recordings
        Design design;
        Carving carving;
        AutoScore score;
        int iteration = 0;
    };

    // Shared with the UI while a run goes.
    struct AutoLive {
        std::mutex mutex;
        int version = 0;                 // bumped whenever `best` changes
        Design best;
        Carving bestCarving;
        AutoScore score;                 // (its target: where the mesh is placed for it)
        int iteration = 0, iterations = 0;
        int rounds = 0, accepted = 0;    // rewrite rounds, and rewrites kept
        std::vector<float> history;      // loss over the run
        std::vector<unsigned char> views;    // RGBA picture of the latest design's silhouettes against the placed mesh's
        int viewsWidth = 0, viewsHeight = 0;
        std::vector<unsigned char> semanticViews;    // RGBA: the semantic loss's views and their targets (when it's on)
        int semanticWidth = 0, semanticHeight = 0;
        std::vector<float> ruleWeights;  // the UI may change these while it runs: each rewrite round reads them (empty: the params')
        int recordFrames = 0;            // set before starting: this many evenly spaced snapshots go into `frames`
        std::vector<AutoFrame> frames;
    };

    struct AutoResult {
        bool ok = false;
        bool cancelled = false;
        Design design;                   // the carving built (a body per piece, a hinge per joint)
        Carving carving;
        AutoScore score;
        std::string message;
    };

    // The grammar's rules (grammar.cpp), with their default weights.
    struct AutoRuleInfo {
        std::string name;
        float weight;
        bool simplifies;             // takes something away (kept when no worse, rather than when better)
    };
    std::vector<AutoRuleInfo> AutoRules();

    // A carving from the sphere up (steps > 0), or (steps == 0) the score of `start` as it is, against the mesh as it is.
    AutoResult AutoDesign(const TriMesh& mesh, const Design& start, const AutoParams& p, JobControl& c,
                          std::shared_ptr<AutoLive> live = nullptr);
}
