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
#include "tracer.hpp"
#include "data/clash.hpp"
#include "data/design.hpp"
#include "data/finish.hpp"
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
        float core = 0.9f;               // the search starts from a shell and a core inside it, this share of the sphere
                                         // across (Carving::ShellAndCore), as far as the shell stays thick enough for a
                                         // hinge's barrel (so by default the shell is as thin as a hinge allows);
                                         // 0: from the whole sphere
        float shellFirst = 0.5f;         // with a core: this share of the steps the core stays shut and only the shell
                                         // is cut (its pieces make the outline); then the core is free too, and "thicken
                                         // a panel" hands its material to the shell's pieces where the shape needs depth
        bool moveTarget = true;          // the mesh it's compared with may be moved, to wherever suits the design best
        bool scaleTarget = true;         // the mesh is scaled (about its centre of volume) to the sphere's volume, so the
                                         // sphere's material fills it exactly; off, it's compared at its own size. The scale
                                         // isn't optimized: a free one let the optimizer shrink the mesh (to ~0.6x) so the
                                         // sphere covered it easily, with ~4x the material it should have
        bool freeScale = false;          // the mesh's scale is optimized too (from the sphere's volume, 0.3x .. 3.3x):
                                         // free, the search can shrink the mesh it's compared with, so the sphere has
                                         // more material than the mesh needs and matching its shape is easier (the
                                         // open design is then bulkier than the mesh, at the mesh's scale)
        int views = 6;                   // directions the silhouettes are compared from
        int viewResolution = 64;         // pixels across each view
        float reachWeight = 1.0f;        // what the mesh's outline being far from the design's costs (0: ignored): each
                                         // missed pixel counts by how far it is from the nearest covered one, so a limb
                                         // nothing is near still pulls on the nearest piece (loss.hpp: Reach)
        float minViewCoverage = 0.6f;    // every view's target must be covered at least this much (steep cost below)
        float collisionWeight = 1.0f;    // what pieces overlapping each other cost (0: ignored) ...
        float collisionRamp = 20.0f;     // ... times this by the end of the search: it rises steadily from the weight to
                                         // this many times it, so the search can try things early on and has to have
                                         // them clear of each other by the end (1: the same throughout)
        float hingeWeight = 2.0f;        // what a hinge being away from its two pieces costs (0: ignored): how far each
                                         // piece is from the axis, over the length a hinge's barrel takes up along it
        float thinWeight = 1.0f;         // what a piece thinner than minThickness costs (0: ignored; thickness.cpp)
        float minThickness = 3.0f;       // mm: the thinnest a piece may be (the largest ball that fits in it, across)
        int hingeChecks = 6;             // poses checked for overlaps in every hinge's turn, spread evenly over its angle
                                         // however long the turn takes (open always is checked; closed needs none: the
                                         // pieces fill the sphere side by side). Each step looks between the last one's.
        bool orderHinges = true;         // the hinges open one after another, in the order that collides least (found
                                         // again at every rewrite round); off: each hinge's timing is Adam's
        // dependencies: every hinge but the one that opens first should be held closed by a piece that moves away
        // before its turn, so the whole design stays shut on one hinge and unfurls by itself once that one is let go
        bool dependencies = false;       // (they need the opening order: orderHinges is taken as on)
        float dependencyWeight = 1.0f;   // what a hinge with nothing near enough to hold it costs
        LedgeParams ledges;              // the ledges that do the holding, added when the search is done
        // finishing, when the search is done (the same passes the editor offers on any design: data/finish.hpp)
        bool buildHinges = true;         // the hinges' hardware (AddHinges). The search then keeps every axis where that
                                         // fits: with hinges.flush, far enough under the surface for the barrel. A piece
                                         // whose hinge can't be built is joined to the one it hung from, loose crumbs
                                         // are cut away, and the result is checked: every piece one solid, on a hinge
        HingeParams hinges;
        bool carveWhenDone = true;       // carve away whatever still overlaps on the way
        CarveParams carve;
        bool latchWhenDone = false;      // a press latch on the hinge that opens first
        LatchParams latch;
        float silhouetteWeight = 1.0f;   // the silhouette comparison's share of the loss (0: only the semantic one)
        // the semantic loss: the views drawn and compared by DINOv3's embeddings (see dino.hpp, semantic.cpp)
        float semanticWeight = 0.0f;     // its share of the loss (0: off)
        std::string semanticModel;       // the .onnx tools/autodesign/export_dinov3.py writes (it sets the views' size)
        std::string onnxRuntime;         // the ONNX Runtime library (empty: the platform's default name, found as usual)
        bool semanticGpu = false;        // run DINOv3 with CUDA, if this ONNX Runtime has it
        SemanticLook semanticLook = SemanticLook::Shaded;
        float patchWeight = 0.0f;        // 0: compare the global (CLS) embeddings .. 1: the patch embeddings, place by place
        int semanticSamples = 4000;      // shaded pixels sampled for the gradient each step (their rest is the edges)
        // the silhouettes drawn on the GPU instead of every core (tracer.hpp); null: on the CPU
        std::shared_ptr<SilhouetteTracer> gpu;
        // the optimizer
        int steps = 6000;                // gradient evaluations in all (the budget), rewrite lookaheads included
        float learningRate = 1.0f;       // Adam's step, in each parameter's own unit (see carving.cpp)
        int stepsPerRound = 60;          // Adam steps between rewrite rounds
        int rewritesPerRound = 3;        // rewrites tried in a round
        int lookahead = 50;              // Adam steps each tried rewrite (and the unchanged design) gets before comparing:
                                         // long enough for a new limb to settle where it helps (with 12 most were
                                         // turned down while still in the way)
        float rewriteMargin = 0.005f;    // a rewrite is kept only if its loss is this much (a share) below the unchanged design's
        int maxPieces = 12;
        std::vector<float> ruleWeights;  // how often each grammar rule is drawn, in AutoRules() order (empty: their defaults)
        unsigned seed = 1;
    };

    struct AutoScore {
        bool valid = false;
        float total = 0.0f;              // the loss: silhouette weight * (fit + coverage penalty) + reach weight * reach
                                         //           + semantic weight * semantic
                                         //           + collision weight * collision + hinge weight * hinges
                                         //           + dependency weight * dependency
                                         //           + thin weight * thin
        float fit = 0.0f;                // open pose against the placed mesh: mean over views of (1 - covered) + extra
        float coveragePenalty = 0.0f;    // 10 per whole share a view is covered less than minViewCoverage
        float viewCover = 0.0f, viewExtra = 0.0f;              // means over the views
        float worstView = 0.0f;          // least covered view
        float reach = 0.0f;              // mean over the views of sum(missed * its distance to the design's outline) /
                                         // sum(T), the distance over 0.1 of the mesh's size
        float collision = 0.0f;          // overlaps: the open pose's + every hinge's turn's (the mean of its poses; loss.cpp)
        float collisionScale = 1.0f;     // what the collision weight was multiplied by when this was scored (the ramp)
        float clashOpen = 0.0f;          // volume inside two pieces at once, open, as a share of the sphere's
        float clashMotion = 0.0f;        // ... at the worst checked pose between
        float clashMotionT = 0.0f;       // ... which is this far through the closing motion
        std::vector<float> checkTimes;   // where in the motion (0 open .. 1 closed) the poses between were checked
        float dependency = 0.0f;         // hinges with nothing to hold them: the mean, over the hinges after the first
                                         // to open, of how far (over the sphere's radius, at most 1) the nearest piece
                                         // that could hold it is from the piece it turns
        int dependents = 0;              // hinges that should be held (the moving ones, less the first to open)
        int holdable = 0;                // ... of them, those with a holder within a ledge's reach
        bool traced = false;             // the silhouettes were drawn on the GPU
        float hinges = 0.0f;             // hinges away from their pieces: per hinge, how far each of its two pieces is
                                         // from its axis (the mean over the barrel's length, beyond a small slack), plus how far it is from fitting under the surface; over the mesh's size
        float hingeGap = 0.0f;           // the farthest a piece is from its hinge's axis, anywhere along the barrel (mm)
        float thin = 0.0f;               // pieces thinner than minThickness: the sum of 1 - thickness / minThickness
        float thinnest = 0.0f;           // the thinnest piece (mm): the largest ball that fits in it, across
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
        Carving carving;                 // (before finishing: ledges, carving and a latch are in the design only)
        AutoScore score;
        MotionOverlap checked;           // the finished design's motion, checked densely for overlaps
        PrintCheck printable;            // with buildHinges: every piece one solid, on a hinge that was built
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
