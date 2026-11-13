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
// Collisions. Pieces overlapping each other in the open pose and motionChecks poses between, placed where joints are
// moving (CheckTimes). Per pose: the depth of the overlap integrated over its volume, relative to the sphere's volume
// times 2% of the mesh's size; the total adds the open pose's and the mean of the poses between. (Closed, the pieces
// fill the sphere side by side: nothing to check.)
//
// Hinges. A joint's pivot should be on (or in) both pieces it joins: how far it is outside each, beyond 1% of the
// mesh's size, adds up over the joints, relative to the mesh's size.
//
// Surface. Hollows should take material from inside the sphere, not its surface: over a thousand points just under the
// surface, the mean of d / (d + 1% of the mesh's size), d how deep the point is inside a hollow. About the share of the
// surface carved away, going up smoothly as hollows reach it.
//
// Shape in 3D (shape.cpp; chamferWeight, sdfWeight). The open design against the placed mesh in 3D: the Chamfer
// distance between their surfaces (both ways), and their signed distances at points in and around the mesh.
//
// Transport (transport.cpp; transportWeight). Optimal transport between the sphere's material, opened (points through
// the sphere, each moving with its piece), and points filling the placed mesh: how far the material has to travel.
// It pulls a piece toward where it's needed even from far away.
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
    // Where in the closing motion (0 open .. 1 closed) the poses between are checked for collisions: only where at
    // least one hinge is moving, half spread evenly over those times and half by how many hinges move at once (bias
    // says how much). n of them, at evenly spaced quantiles, so they move smoothly as the hinges' timing changes.
    std::vector<float> CheckTimes(const Design& d, int n, float bias);

    class Dino;
    class TriangleTree;                  // shape.cpp

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
        // The mesh's scale at which it has the sphere's volume (1 when the sphere is sized from the mesh).
        float VolumeScale() const { return m_MeshVolume > 0.0f ? cbrtf(4.0f / 3.0f * PI * powf(m_SphereRadius, 3.0f) / m_MeshVolume) : 1.0f; }

        // The loss of a carving, against the mesh placed by `target`. With grad: also d loss / d each of ps (the
        // carving's and the target's parameters), per unit of the parameter.
        AutoScore Evaluate(const Carving& c, const Placement& target, const std::vector<Param>& ps = {}, std::vector<float>* grad = nullptr);
        // Any design's (no gradient): its open pose against the mesh, its motion, its hinges.
        AutoScore Evaluate(const Design& d, const Placement& target);
        // The placement kept where the views see all of the mesh, and its scale within 0.3x .. 3.3x.
        Placement KeepTargetInFrame(Placement t) const;

        // Where the last carving scored showed outside the placed mesh: a piece (index) and the point of it there, in
        // the closed pose's frame, and the view's direction there (in that frame too). For aiming new hollows.
        struct Spot {
            int piece;
            Vector3 at, dir;
        };
        const std::vector<Spot>& ExtraSpots() const { return m_Extra; }
        // The sphere's material as transport saw it last (when it's on): a point (closed frame), its piece (index),
        // where the open pose put it and where the plan sends it (world). For aiming new cuts.
        struct Grain {
            int piece;
            Vector3 local, at, to;
        };
        const std::vector<Grain>& Grains() const { return m_Grains; }

        // The silhouettes scored last against their targets, as an RGBA picture.
        void Picture(std::vector<unsigned char>* rgba, int* w, int* h) const;
        // The semantic loss's last views: a row of the placed mesh's, a row of the open design's.
        void SemanticPicture(std::vector<unsigned char>* rgba, int* w, int* h) const;

        float L() const { return m_L; }
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
            std::vector<float> times;            // the poses: 0 (open), then the checks between
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
        // A point of the sphere's material (in the closed pose's frame, which the pieces' fields are in) and the loss's
        // slopes against it: against where its piece's open pose puts it (grad, world), and against its piece's
        // distance there (massSlope: the cutters hand material from piece to piece). See transport.cpp.
        struct Anchor {
            int field;
            Vector3 local;
            Vector3 grad;
            float massSlope;
        };
        using Anchors = std::vector<Anchor>;
        // A point of the design's surface, open: its piece, where it is, and the piece's outward normal there.
        struct DesignPoint {
            int field;
            Vector3 at, normal;
        };

        AutoScore Score(const Design& d, Bodies& b, const Placement& target, Slopes* slopes, Shades* shades);
        void Silhouettes(const Bodies& b, AutoScore& s, Slopes& slopes);
        float FitLoss(const std::vector<std::vector<float>>& target) const;   // the silhouettes' part, against these targets
        void Collisions(const Bodies& b, AutoScore& s, Slopes& slopes);
        void Hinges(const Design& d, const Bodies& b, AutoScore& s) const;
        float PivotGap(const BodyField& f, Vector3 pivot) const { return fmaxf(0.0f, f.Distance(pivot) - 0.01f * m_L); }
        float SurfaceTerm(const Carving& c, float* carved = nullptr) const;
        void Gradient(const Carving& c, const Placement& target, const std::vector<int>& bodyOf, const Bodies& b, const Slopes& slopes,
                      const Shades& shades, const Anchors& anchors, const std::vector<Param>& ps, std::vector<float>& grad);
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
        std::vector<std::vector<float>> SemanticTargets(const Placement& t) const;
        float Look(const View& w, Vector3 p, Vector3 n) const;     // a surface point's shade (world point, normal)
        float DepthLook(float depth) const;                         // the depth look at a depth along the view (from the frame's centre)
        float Background() const;
        Vector3 Normal(const BodyField& f, Vector3 local) const;    // the field's (local) normal
        float ShadeAt(const BodyField& f, const Rigid& xf, const Shade& sh) const;
        // shape.cpp
        void SetUpShape(const TriMesh& mesh);
        void SampleDesignSurface(const Bodies& b);
        float Shape(const Bodies& b, const Placement& t, AutoScore* s, Slopes* slopes) const;   // its weighted share of the loss
        float DesignDistance(const Bodies& b, int pose, Vector3 p, int* who, int skip = -1) const;   // the nearest piece's distance
        // transport.cpp: its weighted share of the loss; with anchors, the material's slopes; and the placement's slopes
        void SetUpTransport(const TriMesh& mesh);
        float Transport(const Carving& c, const Bodies& b, const Placement& t, AutoScore& s, Anchors* anchors, Vector3* gradOffset,
                        float* gradLogScale);
    private:
        AutoParams m_P;
        float m_L = 1.0f;
        Vector3 m_Centre = { 0, 0, 0 };  // of the frame every view shows
        float m_Half = 1.0f;             // the frame is 2·half across
        int m_Res = 64;
        float m_Tau = 1.0f;              // a pixel, in mm: how soft silhouettes are
        std::vector<View> m_Views;
        Vector3 m_MeshCentre = { 0, 0, 0 };          // its centre of volume: placements scale about it
        float m_MeshReach = 1.0f;                     // the farthest the mesh gets from it
        float m_MeshVolume = 0.0f;                    // its volume (0: it isn't closed)
        int m_Fine = 256;                             // the fine drawing's size
        std::vector<std::vector<float>> m_MeshFine;  // the mesh where it is, per view: covered (0/1) at m_Fine
        Placement m_Placed;                           // the placement the targets are drawn for
        std::vector<std::vector<float>> m_Target;    // the placed mesh's silhouette per view
        Vector3 m_SphereCentre = { 0, 0, 0 };
        float m_SphereRadius = 0.0f;
        std::vector<Vector3> m_PairUnit;  // fixed points in [0, 1]^3, mapped into where two bodies' boxes meet
        std::vector<Vector3> m_SurfaceDirs;   // points spread evenly over the sphere (unit directions), for its surface
        std::vector<Spot> m_Extra;
        std::vector<float> m_Cover;       // the last silhouettes (open): views × pixels
        // the 3D shape (shape.cpp)
        std::shared_ptr<TriangleTree> m_Tree;          // the mesh's triangles, for the nearest point of it
        std::vector<Vector3> m_ShapeSurface;           // points on the mesh, where it is ...
        std::vector<Vector3> m_ShapeVolume;            // ... in and around it ...
        std::vector<float> m_ShapeTarget;              // ... and its signed distance at each of those
        std::vector<Vector3> m_DesignUnit;             // fixed points in a unit box, pulled onto each piece's surface
        std::vector<DesignPoint> m_DesignSurface;      // the last evaluated design's surface points
        // transport (transport.cpp)
        std::vector<Vector3> m_BallUnit;               // points through the unit ball: the sphere's material
        std::vector<Vector3> m_MeshFill;               // points filling the mesh, where it is
        std::vector<float> m_TransportG;               // Sinkhorn's mesh potentials, from the last evaluation (a warm start)
        bool m_TransportWarm = false;
        std::vector<Grain> m_Grains;                   // the material as the last evaluation saw it
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
