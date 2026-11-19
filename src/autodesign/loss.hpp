#pragma once
// The loss of a carving (carving.hpp), and its gradient. The closed pose is the sphere whatever the parameters, so only
// the open pose (and the motion between) is scored:
//
// Silhouettes. From each of several directions, the open pose's silhouette is compared with the mesh's, the mesh placed
// by a Placement (an offset and a scale about its centre of volume, which Adam moves too). Per view, with T the
// target's pixel coverage and D the design's:
//   covered = sum(min(T, D)) / sum(T)        extra = sum(max(0, D - T)) / sum(T)
//   loss    = (1 - covered) + extra,         plus 10 per whole share a view is covered less than minViewCoverage
// fit is the mean over the views.
//
// Semantics (when semanticWeight > 0; semantic.cpp). The same views, drawn as pictures (silhouette, depth, or shaded
// grey: SemanticLook) at the size the DINOv3 model was exported for, each compared with the placed mesh's picture by
// DINOv3's embeddings (global, patch, or a mix). DINOv3 gives d loss / d pixel; a pixel depends on the design through
// its soft edges (as silhouettes do) and through the shade of the surface its ray hits.
//
// Collisions. Pieces overlapping each other in the open pose, and in hingeChecks poses of every hinge's turn, spread
// evenly over its angle (data/clash.hpp): a turn is looked at as closely whether it takes all of the motion or a
// twentieth of it, and each evaluation can look between the last one's poses (its phase). Per pose, every two pieces
// that move against each other there are sampled where they can meet; a point inside both costs its depth, and as
// much as 2% of the mesh's size on top once it is deeper than a quarter of that (so shallow overlaps cost steeply
// too, not next to nothing). Over the sphere's volume times 2% of the mesh's size; the total adds the open pose's and
// every hinge's mean over its poses. (Closed, the pieces fill the sphere side by side: nothing to check.)
//
// Dependencies (when asked for). Every hinge after the first to open should have a piece near the one it turns that
// could hold it shut: one that is not carried by the hinge and moves away from the hinge's parent before its turn.
// The term is how far the nearest such piece is, along the sphere's surface.
//
// Hinges. A hinge should be on (or in) both pieces it joins, over the whole length its barrel takes up along the axis
// (PlanHinge), not just at its pivot: at the pivot and half a barrel either way, how far each piece is from the axis,
// beyond half of what a knuckle's leaf can bridge (HingeSlack: 0.6 mm at the default size and reach); the
// mean of the three, added up over the joints, relative to the mesh's size. With buildHinges and hinges.flush, also
// how far the pivot is from being deep enough under the surface for the barrel.
//
// Surface. Hollows should take material from inside the sphere, not its surface: over a thousand points just under the
// surface, the mean of d / (d + 1% of the mesh's size), d how deep the point is inside a hollow. About the share of the
// surface carved away, going up smoothly as hollows reach it.
//
// The silhouettes and collisions depend on the carving only through the pieces' distances at a set of points: the
// silhouettes' at the points where edge pixels' rays come closest to a piece, the collisions' at the points sampled
// inside overlaps. So the gradient is the loss's slope against each such distance (known from the loss), times how the
// distance there changes with each parameter (see Gradient in the .cpp). The placement only moves the targets: its
// slope comes from the targets drawn again with it nudged.
#include "autodesign.hpp"
#include "carving.hpp"
#include "data/field.hpp"
#include <cmath>
#include <memory>
#include <string>

namespace Bakugan {
    class Dino;

    class Loss {
    public:
        Loss(const TriMesh& mesh, const AutoParams& p);
        ~Loss();

        // Why the semantic loss couldn't be set up (empty: it's fine, or not asked for); without it the loss is the rest.
        const std::string& Error() const { return m_Error; }
        bool Semantic() const { return m_Dino != nullptr; }
        std::string SemanticDevice() const;              // where DINOv3 runs ("CPU", "CUDA", or why not CUDA)

        // The sphere a carving starts as: at the mesh's centre of volume, with its volume (or sphereDiameter).
        Vector3 SphereCentre() const { return m_SphereCentre; }
        float SphereRadius() const { return m_SphereRadius; }

        // The loss of a carving, against the mesh placed by `target`. With grad: also d loss / d each of ps (the
        // carving's and the target's parameters), per unit of the parameter.
        // phase (0..1) shifts where in each hinge's turn, and where in space, overlaps are looked for: a different one
        // each step leaves nowhere for an overlap to hide between the checks; the same one compares like with like.
        AutoScore Evaluate(const Carving& c, const Placement& target, const std::vector<Param>& ps = {}, std::vector<float>* grad = nullptr,
                           float phase = 0.5f);
        // Any design's (no gradient): its open pose against the mesh, its motion, its hinges.
        AutoScore Evaluate(const Design& d, const Placement& target, float phase = 0.5f);
        // The order to open an ordered carving's joints in (ids, first opens first). From closed, the joint whose turn
        // overlaps least with everything as it then stands opens next: one that has to wait for another to get out of
        // its way comes after it. With dependencies, among turns that are as free, one that can be held comes first.
        std::vector<int> BestOrder(const Carving& c) const;
        // Multiplies the collision weight from now on (the search raises it as it goes: AutoParams::collisionRamp).
        void SetCollisionScale(float scale) { m_CollisionScale = scale; }
        float CollisionWeight() const { return m_P.collisionWeight * m_CollisionScale; }
        // The placement kept where the views see all of the mesh, and its scale within 0.3x .. 3.3x.
        Placement KeepTargetInFrame(Placement t) const;

        // Where the last carving scored showed outside the placed mesh: a piece (index) and the point of it there, in
        // the closed pose's frame, and the view's direction there (in that frame too). For aiming new hollows.
        struct Spot {
            int piece;
            Vector3 at, dir;
        };
        const std::vector<Spot>& ExtraSpots() const { return m_Extra; }

        // The silhouettes scored last against their targets, as an RGBA picture.
        void Picture(std::vector<unsigned char>* rgba, int* w, int* h) const;
        // The semantic loss's last views: a row of the placed mesh's, a row of the open design's.
        void SemanticPicture(std::vector<unsigned char>* rgba, int* w, int* h) const;
        // A carving's semantic pictures, open (what DINOv3 is shown), drawn `res` across with this loss's views, on the
        // GPU (when given one and it draws: *traced) or the CPU; without DINOv3. For checking the one against the other.
        std::vector<float> SemanticPictures(const Carving& c, int res, bool gpu, bool* traced);

        float L() const { return m_L; }
        // How far from the axis a piece may be and still count as at the hinge: half of what a knuckle's leaf can
        // bridge (so AddHinges can join it with room to spare; a new lid's hinge, along its rim, is about within that).
        float HingeSlack() const { return 0.5f * (m_Hinge.leafLength - m_Hinge.radius); }
        std::vector<Vector3> ViewDirections() const;     // the way each view looks
    private:
        struct View {
            Vector3 dir, u, v;           // looking along dir; u right and v up in the picture
        };
        // A design's bodies (a carving's pieces) in every pose it is scored in.
        struct Bodies {
            std::vector<BodyField> fields;
            std::vector<int> group;              // each field's body (group index)
            std::vector<BoundingBox> box;        // design pose
            std::vector<float> times;            // the poses: 0 (open), then the checks between (then closed, for dependencies)
            std::vector<int> hinge;              // per pose: the hinge (index) whose turn it checks; -1: open; -2: closed
            std::vector<std::vector<Rigid>> xf;  // per pose, per group index
            std::vector<int> fieldOf;            // per group index: its field (-1: none)
        };
        // A point the loss depends on: a body's distance there, in a pose, and the loss's slope against it.
        struct Slope {
            int pose;
            Vector3 at;                          // world
            float weight;                        // d loss / d distance
        };
        using Slopes = std::vector<std::vector<Slope>>;   // per field
        // A semantic pixel's layer: a body its ray reaches (within a pixel's softness), nearest first.
        struct SemLayer {
            int field;
            float c, s, depth;
            float along;                         // d distance / d t at q (< 0: the ray enters there); 0: an edge
            Vector3 at;                          // where the ray comes closest (the edge's point)
            Vector3 q;                           // the point shaded
        };
        // A surface point a semantic pixel's shade comes from, and the loss's slope against that shade. Moved with
        // the body, the point slides along the pixel's ray to stay on the surface (to first order), and is shaded anew.
        struct Shade {
            int pose, view;
            Vector3 at;                          // world: where the ray hits the body (or, on an edge, comes closest)
            float along;                         // d distance / d t along the ray there (< 0: it enters); 0: an edge
            float weight;                        // d loss / d shade
        };
        using Shades = std::vector<std::vector<Shade>>;   // per field

        AutoScore Score(const Design& d, Bodies& b, const Placement& target, Slopes* slopes, Shades* shades, float phase);
        void Silhouettes(const Bodies& b, AutoScore& s, Slopes& slopes);
        // The open pose drawn: per pixel the piece that covers it most (-1: none), its coverage, and where its ray
        // comes closest to it. On the GPU when there is a tracer (and it works), on every core otherwise.
        void Draw(const Bodies& b, std::vector<int>& who, std::vector<Vector3>& at, bool* traced);
        bool DrawTraced(const Bodies& b, std::vector<int>& who, std::vector<Vector3>& at);
        Vector3 RayStart(const BoundingBox& box, const View& w, float sx, float sy) const;
        // the silhouettes' part (fit + coverage penalty), against these targets (and these depths, with FitLook::Depth)
        float FitLoss(const std::vector<std::vector<float>>& target, const std::vector<std::vector<float>>* depth = nullptr) const;
        bool DepthFit() const { return m_P.fitLook == FitLook::Depth; }
        // The placed mesh's front depth per pixel (along the view, from the frame's centre; where it covers nothing: 0)
        std::vector<std::vector<float>> TargetDepths(const Placement& t) const;
        float DepthCost(float dz, float* slope) const;   // a covered pixel's depth mismatch cost, and its slope
        void OnePiece(const Bodies& b, AutoScore& s, Slopes& slopes);   // the one-piece term (AutoParams::onePiece)
        void Collisions(const Design& d, const Bodies& b, AutoScore& s, Slopes& slopes, float phase);
        // What one sample inside two pieces costs, and its slope against its depth.
        float ClashCost(float depth, float* slope) const;
        // Which piece (field) every point just under the sphere's surface belongs to (-1: none, a hollow).
        std::vector<int> SurfaceOwners(const Bodies& b) const;
        // The nearest piece that could hold hinge h shut, given the hinges that open before it: how far along the
        // surface (mm; FLT_MAX: there is none), and the two surface points (indices) it is measured between.
        float HoldGap(const Design& d, const Bodies& b, const std::vector<std::vector<char>>& under, const std::vector<int>& owner, int h,
                      const std::vector<int>& earlier, int* from = nullptr, int* to = nullptr) const;
        void Dependencies(const Design& d, const Bodies& b, AutoScore& s, Slopes& slopes);
        float SurfaceSpacing() const;
        void Hinges(const Design& d, const Bodies& b, AutoScore& s) const;
        // One hinge against its two pieces (see Hinges); farthest: the most either is from the axis along the barrel.
        float HingeCost(const Hinge& h, const BodyField& parent, const BodyField& child, float* farthest = nullptr) const;
        float HingeWeight() const { return m_P.hingeWeight; }
        float SurfaceTerm(const Carving& c, float* carved = nullptr) const;
        float SurfaceTermWith(const Carving& c, const std::vector<float>& others, const SuperFrustum& changed) const;
        bool Gradient(const Carving& c, const Placement& target, const std::vector<int>& bodyOf, const Bodies& b, const Slopes& slopes,
                      const Shades& shades, const std::vector<Param>& ps, std::vector<float>& grad, AutoScore* s = nullptr);
        // A ray through the frame at (sx, sy) (from its centre, in mm) against one body (moved by xf), sphere traced
        // through its field inside its box; tau: the pixel size. Returns the nearest it gets (negative: how deep), and
        // where (world); with hit, also where it first enters the body (world; only when it does: the return is < 0).
        float Trace(const BodyField& f, const BoundingBox& box, const Rigid& xf, const View& w, float sx, float sy, float tau,
                    Vector3* at, Vector3* hit = nullptr) const;
        float FrameX(int x, int res) const { return ((x + 0.5f) / res * 2.0f - 1.0f) * m_Half; }
        float FrameY(int y, int res) const { return (1.0f - (y + 0.5f) / res * 2.0f) * m_Half; }
        // The placed mesh's silhouettes (res = m_Res), from a fine drawing of the mesh where it is (bilinear, 2x2 a pixel).
        std::vector<std::vector<float>> TargetSilhouettes(const Placement& t) const;
        // A point of the frame (sx, sy) of view w, where it was before the mesh was placed by t.
        Vector2 Unplace(const View& w, float sx, float sy, const Placement& t) const;
        // semantic.cpp
        void SetUpSemantic(const TriMesh& mesh);
        void Semantics(const Bodies& b, AutoScore& s, Slopes& slopes, Shades& shades);
        // The open design's semantic pictures (into m_SemDrawn) and every pixel's layers (MaxLayers a pixel, count
        // of them used): the rays traced on the GPU when there is one and it draws (then true), else on the CPU.
        bool DrawSemantic(const Bodies& b, std::vector<SemLayer>& layers, std::vector<unsigned char>& count, bool useGpu);
        std::vector<std::vector<float>> SemanticTargets(const Placement& t) const;
        float Look(const View& w, Vector3 p, Vector3 n) const;     // a surface point's shade (world point, normal)
        float DepthLook(float depth) const;                         // the depth look at a depth along the view (from the frame's centre)
        float Background() const;
        Vector3 Normal(const BodyField& f, Vector3 local) const;    // the field's (local) normal
        float ShadeAt(const BodyField& f, const Rigid& xf, const Shade& sh) const;
    private:
        AutoParams m_P;
        float m_CollisionScale = 1.0f;
        float m_L = 1.0f;
        Vector3 m_Centre = { 0, 0, 0 };  // of the frame every view shows
        float m_Half = 1.0f;             // the frame is 2·half across
        int m_Res = 64;
        float m_Tau = 1.0f;              // a pixel, in mm: how soft silhouettes are
        std::vector<View> m_Views;
        Vector3 m_MeshCentre = { 0, 0, 0 };          // its centre of volume: placements scale about it
        float m_MeshReach = 1.0f;                     // the farthest the mesh gets from it
        int m_Fine = 256;                             // the fine drawing's size
        std::vector<std::vector<float>> m_MeshFine;  // the mesh where it is, per view: covered (0/1) at m_Fine
        std::vector<std::vector<float>> m_MeshFineZ; // ... (FitLook::Depth) covered × its front depth there
        std::vector<std::vector<float>> m_TargetDepth;   // the placed mesh's front depth per view (FitLook::Depth)
        Placement m_Placed;                           // the placement the targets are drawn for
        std::vector<std::vector<float>> m_Target;    // the placed mesh's silhouette per view
        Vector3 m_SphereCentre = { 0, 0, 0 };
        float m_SphereRadius = 0.0f;
        HingeSizes m_Hinge;              // the hardware a hinge would get (data/finish.hpp)
        std::vector<Vector3> m_PairUnit;  // fixed points in [0, 1]^3, mapped into where two bodies' boxes meet
        std::vector<Vector3> m_SurfaceDirs;   // points spread evenly over the sphere (unit directions), for its surface
        std::vector<Spot> m_Extra;
        std::vector<float> m_Cover;       // the last silhouettes (open): views × pixels
        // FitLook::Depth: per pixel, the pieces its ray enters (up to 4, nearest first), each with an opacity that rises from
        // 0 at its silhouette's edge to 1 a pixel inside (alpha = clamp(-nearest / tau, 0, 1)), composited front to back:
        //   w_i = alpha_i · prod_{j<i} (1 - alpha_j),   A = sum w_i,   z = sum w_i z_i / A
        // so the depth never jumps: not where a piece's edge comes or goes, nor where a near piece's edge crosses a far one.
        struct DepthLayer {
            int pixel, field;
            float z, alpha;
            Vector3 hit;                  // where the ray enters it (world)
            Vector3 at;                   // where the ray comes nearest it (its edge's point)
        };
        std::vector<float> m_Z;           // the composited depth per pixel (views × pixels)
        std::vector<float> m_ZA;          // ... and its opacity A (0: nothing entered)
        std::vector<DepthLayer> m_ZLayers;   // every pixel's layers, a pixel's together, nearest first
        // a pixel's layers (any order, and how many) made its composite, kept in row (nearest 4 first)
        void CompositeDepth(size_t pixel, std::vector<DepthLayer>& layers, std::vector<DepthLayer>& row);
        bool DrawTracedDepth(const Bodies& b, std::vector<int>& who, std::vector<Vector3>& at);   // FitLook::Depth on the GPU
        // the semantic loss
        std::string m_Error;
        std::unique_ptr<Dino> m_Dino;
        int m_SemRes = 0;                 // its views' size (the model's)
        float m_SemTau = 1.0f;            // its pixel, in mm
        int m_SemFine = 0;                // the mesh's fine drawing for it: m_SemFine across
        std::vector<std::vector<float>> m_SemHit, m_SemDepth, m_SemShade;   // per view: covered, × depth from its centre, × shade
        std::vector<std::vector<float>> m_SemTarget;  // the placed mesh's picture per view (for m_Placed)
        std::vector<float> m_SemDrawn;    // the last pictures (open): views × pixels
        std::vector<float> m_SemGradTarget;  // and the loss's slope against each target pixel
    };
}
