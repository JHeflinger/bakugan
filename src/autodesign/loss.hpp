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
// The first two parts depend on the design only through the bodies' distances at a set of points: the silhouettes' at the
// points where edge pixels' rays come closest to the body, the collisions' at the points sampled inside overlaps.
// So the gradient is the loss's slope against each such distance (known from the loss), times how the distance
// there changes with each parameter (see Gradient in the .cpp).
#include "autodesign.hpp"
#include "params.hpp"
#include "data/field.hpp"
#include <cmath>

namespace Bakugan {
    // Where in the closing motion (0 open .. 1 closed) the poses between are checked for collisions: only where at
    // least one hinge is moving, half spread evenly over those times and half by how many hinges move at once (bias
    // says how much). n of them, at evenly spaced quantiles, so they move smoothly as the hinges' timing changes.
    std::vector<float> CheckTimes(const Design& d, int n, float bias);

    class Loss {
    public:
        Loss(const TriMesh& mesh, const AutoParams& p);

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

        void Silhouettes(const Bodies& b, AutoScore& s, Slopes& slopes);
        void Collisions(const Bodies& b, AutoScore& s, Slopes& slopes);
        void Hinges(const Design& d, const Bodies& b, AutoScore& s) const;
        float PivotGap(const BodyField& f, Vector3 pivot) const { return fmaxf(0.0f, f.Distance(pivot) - 0.01f * m_L); }
        void Gradient(const Design& d, const Bodies& b, const Slopes& slopes, const std::vector<Param>& ps, std::vector<float>& grad) const;
        float Trace(const BodyField& f, const BoundingBox& box, const Rigid& xf, const View& w, int x, int y, Vector3* at) const;
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
    };
}
