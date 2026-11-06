#pragma once
// The loss, and its gradient. Two parts:
//
// Silhouettes. From each of several directions, the open pose's silhouette is compared with the mesh's and the closed
// pose's with a disc the size of the sphere. Per view, with T the target's pixel coverage and D the design's:
//   covered = sum(min(T, D)) / sum(T)        extra = sum(max(0, D - T)) / sum(T)
//   loss    = (1 - covered) + extra,         plus 10 per whole share a view is covered less than minViewCoverage
// fit and sphere are the means over the views; the total weighs them by the goal (sphereBias).
//
// Collisions. Bodies overlapping each other in the open pose, the closed pose, and motionChecks poses between, placed
// where hinges are moving (CheckTimes). Per pose: the depth of the overlap integrated over its volume, relative to
// the sphere's volume times 2% of the mesh's size; the total adds open + closed + the mean of the poses between.
//
// Both parts depend on the design only through the bodies' distances at a set of points: the silhouettes' at the
// points where edge pixels' rays come closest to the body, the collisions' at the points sampled inside overlaps.
// So the gradient is the loss's slope against each such distance (known from the loss), times how the distance
// there changes with each parameter (see Gradient in the .cpp).
#include "autodesign.hpp"
#include "params.hpp"
#include "data/field.hpp"

namespace Bakugan {
    // Where in the closing motion (0 open .. 1 closed) the poses between are checked for collisions: only where at
    // least one hinge is moving, half spread evenly over those times and half by how many hinges move at once (bias
    // says how much). n of them, at evenly spaced quantiles, so they move smoothly as the hinges' timing changes.
    std::vector<float> CheckTimes(const Design& d, int n, float bias);

    class Loss {
    public:
        Loss(const TriMesh& mesh, const AutoParams& p);

        // The loss of a design. With grad: also d loss / d parameter for each of ps, per unit of the parameter.
        AutoScore Evaluate(const Design& d, const std::vector<Param>& ps, std::vector<float>* grad);

        // The silhouettes scored last against their targets, as an RGBA picture (open pose above, closed below).
        void Picture(std::vector<unsigned char>* rgba, int* w, int* h) const;

        float L() const { return m_L; }
        const std::vector<Vector3>& Surface() const { return m_Surface; }   // points on the mesh
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
        };
        // A point the loss depends on: a body's distance there, in a pose, and the loss's slope against it.
        struct Slope {
            int pose;
            Vector3 at;                          // world
            float weight;                        // d loss / d distance
        };
        using Slopes = std::vector<std::vector<Slope>>;   // per field

        void Silhouettes(const Bodies& b, AutoScore& s, Slopes& slopes);
        void Collisions(const Bodies& b, AutoScore& s, Slopes& slopes) const;
        void Gradient(const Design& d, const Bodies& b, const Slopes& slopes, const std::vector<Param>& ps, std::vector<float>& grad) const;
        float Trace(const BodyField& f, const BoundingBox& box, const Rigid& xf, const View& w, int x, int y, Vector3* at) const;
        Vector2 ToPixel(const View& w, Vector3 p) const;
        float Disc(const View& w, int x, int y) const;   // the sphere's silhouette
    private:
        AutoParams m_P;
        float m_L = 1.0f;
        Vector3 m_Centre = { 0, 0, 0 };  // of the frame every view shows
        float m_Half = 1.0f;             // the frame is 2·half across
        int m_Res = 64;
        float m_Tau = 1.0f;              // a pixel, in mm: how soft silhouettes are
        std::vector<View> m_Views;
        std::vector<std::vector<float>> m_Mesh;     // the mesh's silhouette per view
        Vector3 m_SphereCentre = { 0, 0, 0 };
        float m_SphereRadius = 0.0f;
        std::vector<Vector3> m_Surface;
        std::vector<Vector3> m_PairUnit;  // fixed points in [0, 1]^3, mapped into where two bodies' boxes meet
        std::vector<float> m_Cover[2];    // the last silhouettes (open, closed): views × pixels
    };
}
