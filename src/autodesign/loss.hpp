#pragma once
// The loss, and its gradient. Two parts:
//
// Silhouettes. From each of several directions, the open pose's silhouette is compared with the mesh's and the closed
// pose's with a disc the size of the sphere. The sphere's centre is free (the optimizer moves it, see Evaluate): the
// closed design can be a sphere anywhere, as long as it is one. Per view, with T the target's pixel coverage and D the design's:
//   covered = sum(min(T, D)) / sum(T)        extra = sum(max(0, D - T)) / sum(T)
//   loss    = (1 - covered) + extra,         plus 10 per whole share a view is covered less than minViewCoverage
// fit and sphere are the means over the views; the total weighs them by the goal (sphereBias).
//
// Collisions. Bodies overlapping each other in the open pose, the closed pose, and motionChecks poses between, placed
// where hinges are moving (CheckTimes). Per pose: the depth of the overlap integrated over its volume, relative to
// the sphere's volume times 2% of the mesh's size; the total adds open + closed + the mean of the poses between.
//
// Hinges. A hinge's pivot should be on (or in) both bodies it joins: how far it is outside each, beyond 1% of the
// mesh's size, adds up over the hinges, relative to the mesh's size.
//
// Semantics (when semanticWeight > 0; semantic.cpp). The same views, drawn as pictures (silhouette, depth, or shaded
// grey: SemanticLook) at the size the DINOv3 model was exported for, each compared with the mesh's (open) or the
// sphere's (closed) picture by DINOv3's embeddings (global, patch, or a mix). DINOv3 gives d loss / d pixel; a pixel
// depends on the design through its soft edges (as silhouettes do) and through the shade of the surface its ray hits.
//
// The first two parts depend on the design only through the bodies' distances at a set of points: the silhouettes' at the
// points where edge pixels' rays come closest to the body, the collisions' at the points sampled inside overlaps.
// So the gradient is the loss's slope against each such distance (known from the loss), times how the distance
// there changes with each parameter (see Gradient in the .cpp).
#include "autodesign.hpp"
#include "params.hpp"
#include "data/field.hpp"
#include <cmath>
#include <memory>
#include <string>

namespace Bakugan {
    // Where in the closing motion (0 open .. 1 closed) the poses between are checked for collisions: only where at
    // least one hinge is moving, half spread evenly over those times and half by how many hinges move at once (bias
    // says how much). n of them, at evenly spaced quantiles, so they move smoothly as the hinges' timing changes.
    std::vector<float> CheckTimes(const Design& d, int n, float bias);

    class Dino;

    class Loss {
    public:
        Loss(const TriMesh& mesh, const AutoParams& p);
        ~Loss();

        // Why the semantic loss couldn't be set up (empty: it's fine, or not asked for); without it the loss is the rest.
        const std::string& Error() const { return m_Error; }
        bool Semantic() const { return m_Dino != nullptr; }
        std::string SemanticDevice() const;              // where DINOv3 runs ("CPU", "CUDA", or why not CUDA)

        // The loss of a design against the sphere centred at `sphere`. With grad: also d loss / d parameter for each of
        // ps, per unit of the parameter; with sphereGrad, d loss / d the sphere's centre, per SphereUnit().
        AutoScore Evaluate(const Design& d, Vector3 sphere, const std::vector<Param>& ps, std::vector<float>* grad,
                           Vector3* sphereGrad = nullptr);

        // The sphere: where it starts (the mesh's centre of volume), how far it may go (it stays inside the frame the
        // views show, or its silhouettes would be cut off), and the size of a typical step of its centre.
        Vector3 SphereStart() const { return m_SphereStart; }
        std::vector<Vector3> ViewDirections() const;     // the way each view looks
        Vector3 KeepSphereInFrame(Vector3 c) const;
        float SphereUnit() const { return 0.005f * m_L; }
        // The best place for the sphere around a design that stays as it is (Adam on its centre alone; for scoring a
        // finished design, whose sphere wasn't optimized with it).
        Vector3 FitSphere(const Design& d);

        // The silhouettes scored last against their targets, as an RGBA picture (open pose above, closed below).
        void Picture(std::vector<unsigned char>* rgba, int* w, int* h) const;
        // The semantic loss's last views: rows of the mesh's, the open design's, the sphere's, the closed design's.
        void SemanticPicture(std::vector<unsigned char>* rgba, int* w, int* h) const;

        float L() const { return m_L; }
        const std::vector<Vector3>& Surface() const { return m_Surface; }   // points on the mesh

        // Where the last evaluated design was wrong, for the grammar to aim at: a body (group index), a point in its
        // design pose frame, and (Extra) the direction its view looked along, in that frame.
        struct Spot {
            int group;
            Vector3 at, dir;
        };
        const std::vector<Spot>& ExtraSpots() const { return m_Extra; }        // open pose: design pixels where the mesh has none
        const std::vector<Spot>& OverlapSpots() const { return m_Overlaps; }   // inside two bodies, in any checked pose
    private:
        struct View {
            Vector3 dir, u, v;           // looking along dir; u right and v up in the picture
        };
        // A design's bodies in every pose it is scored in.
        struct Bodies {
            std::vector<BodyField> fields;
            std::vector<int> group;              // each field's body (group index)
            std::vector<BoundingBox> box;        // design pose
            std::vector<float> times;            // the poses: 0 (open), 1 (closed), then the checks between
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
        // A surface point a semantic pixel's shade comes from, and the loss's slope against that shade. Moved with
        // the body, the point slides along the pixel's ray to stay on the surface (to first order), and is shaded anew.
        struct Shade {
            int pose, view;
            Vector3 at;                          // world: where the ray hits the body (or, on an edge, comes closest)
            float along;                         // d distance / d t along the ray there (< 0: it enters); 0: an edge
            float weight;                        // d loss / d shade
        };
        using Shades = std::vector<std::vector<Shade>>;   // per field

        void Silhouettes(const Bodies& b, AutoScore& s, Slopes& slopes);
        void Collisions(const Bodies& b, AutoScore& s, Slopes& slopes);
        void Hinges(const Design& d, const Bodies& b, AutoScore& s) const;
        float PivotGap(const BodyField& f, Vector3 pivot) const { return fmaxf(0.0f, f.Distance(pivot) - 0.01f * m_L); }
        void Gradient(const Design& d, const Bodies& b, const Slopes& slopes, const Shades& shades, const std::vector<Param>& ps,
                      std::vector<float>& grad) const;
        // A ray through the frame at (sx, sy) (from its centre, in mm) against one body (moved by xf), sphere traced
        // through its field inside its box; tau: the pixel size. Returns the nearest it gets (negative: how deep), and
        // where (world); with hit, also where it first enters the body (world; only when it does: the return is < 0).
        float Trace(const BodyField& f, const BoundingBox& box, const Rigid& xf, const View& w, float sx, float sy, float tau,
                    Vector3* at, Vector3* hit = nullptr) const;
        float FrameX(int x, int res) const { return ((x + 0.5f) / res * 2.0f - 1.0f) * m_Half; }
        float FrameY(int y, int res) const { return (1.0f - (y + 0.5f) / res * 2.0f) * m_Half; }
        // semantic.cpp
        void SetUpSemantic(const TriMesh& mesh);
        void Semantics(const Bodies& b, AutoScore& s, Slopes& slopes, Shades& shades);
        float Look(const View& w, Vector3 p, Vector3 n) const;     // a surface point's shade (world point, normal)
        float Background() const;
        Vector3 Normal(const BodyField& f, Vector3 local) const;    // the field's (local) normal
        float ShadeAt(const BodyField& f, const Rigid& xf, const Shade& sh) const;
        void SphereLook(const View& w, Vector3 centre, float* out) const;
        Vector2 ToPixel(const View& w, Vector3 p) const;
        float Disc(const View& w, int x, int y, Vector3 centre) const;     // the sphere's silhouette
        float ClosedLoss(Vector3 centre) const;          // the closed pose's part of the loss, from the last silhouettes
    private:
        AutoParams m_P;
        float m_L = 1.0f;
        Vector3 m_Centre = { 0, 0, 0 };  // of the frame every view shows
        float m_Half = 1.0f;             // the frame is 2·half across
        int m_Res = 64;
        float m_Tau = 1.0f;              // a pixel, in mm: how soft silhouettes are
        std::vector<View> m_Views;
        std::vector<std::vector<float>> m_Mesh;     // the mesh's silhouette per view
        Vector3 m_SphereStart = { 0, 0, 0 };
        Vector3 m_SphereCentre = { 0, 0, 0 };             // of the design scored last
        float m_SphereRadius = 0.0f;
        std::vector<Vector3> m_Surface;
        std::vector<Vector3> m_PairUnit;  // fixed points in [0, 1]^3, mapped into where two bodies' boxes meet
        std::vector<float> m_Cover[2];    // the last silhouettes (open, closed): views × pixels
        std::vector<Spot> m_Extra, m_Overlaps;
        // the semantic loss
        std::string m_Error;
        std::unique_ptr<Dino> m_Dino;
        int m_SemRes = 0;                 // its views' size (the model's)
        float m_SemTau = 1.0f;            // its pixel, in mm
        std::vector<std::vector<float>> m_MeshLook;   // the mesh's picture per view
        std::vector<float> m_SemDrawn[2], m_SemTarget[2];   // the last pictures (open, closed): views × pixels
        Vector3 m_SemSphereGrad = { 0, 0, 0 };        // its d loss / d the sphere's centre (per mm), from the last evaluation
    };
}
