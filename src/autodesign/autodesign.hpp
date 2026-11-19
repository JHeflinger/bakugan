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

    struct AutoStart;

    // How the open design's views are compared with the mesh's (the fit):
    enum class FitLook {
        Silhouette,                      // outlines only: covered and extra pixels
        Depth,                           // outlines, and where both cover a pixel, how far apart their front surfaces are
                                         // along the view (a depth raster each): the views see the shape in 3D, not just
                                         // its outline
    };

    struct AutoParams {
        // what is wanted
        float sphereDiameter = 0.0f;     // mm; 0: a sphere of the mesh's volume (it sits at the mesh's centre of volume)
        // where the search starts: null, the sphere whole; otherwise a design the generator made (a saved .txt, or the
        // editor's), carried on from (resume.cpp)
        std::shared_ptr<const AutoStart> startFrom;
        bool moveTarget = true;          // the mesh it's compared with may be moved, to wherever suits the design best
        bool scaleTarget = true;         // ... and scaled (0.3x .. 3.3x, about its centre of volume)
        // every body one piece: the volume of what lies apart from each body's largest piece (seen in voxels of
        // onePieceVoxels to the sphere's radius) costs onePieceWeight per sphere's volume, and its gradient shrinks those
        // parts away; when the search ends, what is still apart is taken away (RemoveFloatingBits: all of it)
        bool onePiece = true;
        float onePieceWeight = 10.0f;
        int onePieceVoxels = 24;
        FitLook fitLook = FitLook::Silhouette;
        float depthScale = 0.05f;        // FitLook::Depth: a depth mismatch this big (a share of the mesh's size) costs a
                                         // covered pixel half as much as a missing one; more costs up to as much (robust:
                                         // d² / (d² + scale²)), so a piece far in front of the mesh isn't punished forever
        int views = 32;                   // directions the silhouettes are compared from
        int viewResolution = 1080;         // pixels across each view
        float minViewCoverage = 0.6f;    // every view's target must be covered at least this much (steep cost below)
        float collisionWeight = 1.0f;    // what pieces overlapping each other cost (0: ignored) ...
        float collisionRamp = 20.0f;     // ... times this by the end of the search: it rises steadily from the weight to
                                         // this many times it, so the search can try things early on and has to have
                                         // them clear of each other by the end (1: the same throughout)
        float hingeWeight = 2.0f;        // what a hinge being away from its two pieces costs (0: ignored): how far each
                                         // piece is from the axis, over the length a hinge's barrel takes up along it
        float surfaceWeight = 1.0f;      // what hollows carving into the sphere's surface cost (0: ignored)
        int hingeChecks = 32;             // poses checked for overlaps in every hinge's turn, spread evenly over its angle
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
        bool buildHinges = false;        // the hinges' hardware (AddHinges). The search then keeps every axis where that
                                         // fits: with hinges.flush, far enough under the surface for the barrel
        HingeParams hinges;
        bool carveWhenDone = false;      // carve away whatever still overlaps on the way
        CarveParams carve;
        bool latchWhenDone = false;      // a press latch on the hinge that opens first
        LatchParams latch;
        bool removeFloatingWhenDone = false;   // last: take away bits of bodies that float apart from the rest (slivers
        FloatingParams floating;               // many cuts leave), smaller than floating.share of the body's largest piece
        float silhouetteWeight = 1.0f;   // the silhouette comparison's share of the loss (0: only the semantic one)
        // the semantic loss: the views drawn and compared by DINOv3's embeddings (see dino.hpp, semantic.cpp)
        float semanticWeight = 0.0f;     // its share of the loss (0: off)
        std::string semanticModel;       // the .onnx tools/autodesign/export_dinov3.py writes (it sets the views' size)
        std::string onnxRuntime;         // the ONNX Runtime library (empty: the platform's default name, found as usual)
        bool semanticGpu = false;        // run DINOv3 with CUDA, if this ONNX Runtime has it
        SemanticLook semanticLook = SemanticLook::Shaded;
        float patchWeight = 0.0f;        // 0: compare the global (CLS) embeddings .. 1: the patch embeddings, place by place
        // A video of the semantic comparison over the run (the mesh's views DINOv3 sees against the design's): written
        // to this .mp4 (needs ffmpeg; empty: none). A frame every semanticVideoEvery steps of the search (its lookahead
        // tries aside), then the best design held for two seconds.
        std::string semanticVideo;
        int semanticVideoEvery = 5;
        int semanticVideoFps = 15;
        int semanticSamples = 4000;      // shaded pixels sampled for the gradient each step (their rest is the edges)
        // the silhouettes drawn on the GPU instead of every core (tracer.hpp); null: on the CPU
        std::shared_ptr<SilhouetteTracer> gpu;
        // the optimizer
        int steps = 1500;                // gradient evaluations in all (the budget), rewrite lookaheads included
        float learningRate = 1.0f;       // Adam's step, in each parameter's own unit (see carving.cpp)
        int stepsPerRound = 60;          // Adam steps between rewrite rounds
        int rewritesPerRound = 3;        // rewrites tried in a round
        int lookahead = 12;              // Adam steps each tried rewrite (and the unchanged design) gets before comparing
        float rewriteMargin = 0.005f;    // a rewrite is kept only if its loss is this much (a share) below the unchanged design's
        int maxPieces = 64;
        int maxHollows = 64;
        std::vector<float> ruleWeights;  // how often each grammar rule is drawn, in AutoRules() order (empty: their defaults)
        unsigned seed = 1;
    };

    struct AutoScore {
        bool valid = false;
        float total = 0.0f;              // the loss: silhouette weight * (fit + coverage penalty) + semantic weight * semantic
                                         //           + collision weight * collision + hinge weight * hinges
                                         //           + surface weight * surface + dependency weight * dependency
        float fit = 0.0f;                // open pose against the placed mesh: mean over views of (1 - covered) + extra
                                         // (+ depth, with FitLook::Depth)
        float fragments = 0.0f;          // one piece: the volume apart from each body's largest piece, over the sphere's
        int fragmentParts = 0;           // ... how many parts that is
        float depth = 0.0f;              // FitLook::Depth: mean over views of the depth mismatch, sum over pixels both
                                         // cover of min(T, D) · d² / (d² + scale²), over sum(T)
        float coveragePenalty = 0.0f;    // 10 per whole share a view is covered less than minViewCoverage
        float viewCover = 0.0f, viewExtra = 0.0f;              // means over the views
        float worstView = 0.0f;          // least covered view
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
        bool gradientTraced = false;     // ... and their gradient's field evaluations made there
        // where the step's time went (ms): drawing and comparing the views, the semantic loss (drawing, DINOv3, its
        // slopes), collisions, the gradient's preparing (every parameter nudged: the pieces rebuilt; shades, hinges) and
        // measuring (every slope point measured again), and the placement's (its targets drawn again)
        float msViews = 0, msSemantic = 0, msCollisions = 0, msPrepare = 0, msMeasure = 0, msPlacement = 0, msOnePiece = 0;
        bool semanticTraced = false;     // ... and the semantic pictures
        float hinges = 0.0f;             // hinges away from their pieces: per hinge, how far each of its two pieces is
                                         // from its axis (the mean over the barrel's length, beyond a small slack), plus how far it is from fitting under the surface; over the mesh's size
        float hingeGap = 0.0f;           // the farthest a piece is from its hinge's axis, anywhere along the barrel (mm)
        float surface = 0.0f;            // hollows into the sphere's surface: over points just under it, the mean of
                                         // d / (d + 1% of the mesh's size), d how deep inside a hollow it is
        float surfaceCarved = 0.0f;      // the share of those points inside a hollow
        float semantic = 0.0f;           // DINOv3: mean over the views of the open pose's loss against the placed mesh
        float worstSemantic = 0.0f;      // the worst single view's
        Placement target;                // where the mesh was placed when this was scored
        Vector3 centre = { 0, 0, 0 };    // the sphere's (the closed pose)
        float radius = 0.0f;
        float meshSize = 0.0f;           // the mesh's size (the loss's L), in the frame it was scored in
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
        std::string devices;             // where the latest step ran: the views drawn on the GPU or here, DINOv3 on CUDA or not
        bool deviceShortfall = false;    // ... and something asked for the GPU didn't get it
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
        std::string semanticVideo;       // the semantic video's path, when one was written
        std::string message;
    };

    // The grammar's rules (grammar.cpp), with their default weights.
    struct AutoRuleInfo {
        std::string name;
        float weight;
        bool simplifies;             // takes something away (kept when no worse, rather than when better)
    };
    std::vector<AutoRuleInfo> AutoRules();

    // A carving from the sphere up (steps > 0; or from p.startFrom), or (steps == 0) the score of `start` as it is,
    // against the mesh as it is.
    AutoResult AutoDesign(const TriMesh& mesh, const Design& start, const AutoParams& p, JobControl& c,
                          std::shared_ptr<AutoLive> live = nullptr);

    // ---- starting from a design (resume.cpp) -------------------------------------------------------------------------
    // A design the generator made, to carry the search on from, and (when it was saved with one) where the mesh was
    // placed for it.
    struct AutoStart {
        Design design;
        std::string source;              // what it is, for messages ("bunny.txt", "the current design")
        bool hasTarget = false;
        Placement target;                // the mesh's placement it was scored with ...
        float meshSize = 0.0f;           // ... and the mesh's size then (0: unknown), so another frame can be mapped
    };

    // A design's text with the placement it was scored with: Design::Serialize, plus an "autodesign-target" line that
    // Design::Deserialize (and older builds) skip. Only the design when the score isn't valid.
    std::string SerializeWithTarget(const Design& d, const AutoScore& score);
    // Reads such a text (or any design file: then without a placement).
    bool ParseAutoStart(const std::string& text, const std::string& source, AutoStart* out, std::string* err);
    bool ReadAutoStart(const std::string& path, AutoStart* out, std::string* err);
    // The carving a design was built from (Carving::Build, the other way round): bodies with the generator's sphere,
    // "Cutter N" / "Hollow N" cuts and "Outside cutter N" features, hinges between them. Whatever finishing added
    // (hinge hardware, ledges, clearances, a latch) is left out, and listed in `notes`. False, and why, for a design the
    // generator didn't make (or one edited past recognising).
    bool CarvingFromDesign(const Design& d, Carving* out, std::string* err, std::string* notes = nullptr);
}
