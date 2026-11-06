#pragma once
// The loss: silhouettes of the design against silhouettes of the mesh and of a sphere, and its gradient.
//
// From each of several directions, the open pose's silhouette is compared with the mesh's and the closed pose's with
// a disc the size of the sphere. Per view, with T the target's pixel coverage and D the design's:
//   covered = sum(min(T, D)) / sum(T)        extra = sum(max(0, D - T)) / sum(T)
//   loss    = (1 - covered) + extra,         plus 10 per whole share a view is covered less than minViewCoverage
// fit and sphere are the means over the views; the total weighs them by the goal (sphereBias).
//
// Silhouettes are soft: a pixel's coverage comes from how close its ray gets to the body (0 half a pixel outside, 1
// half a pixel inside), so they change smoothly with the design. That makes the gradient cheap (see Loss in the .cpp):
// only pixels on the edge of a silhouette depend on the parameters, each through the body's distance at the one
// point where its ray came closest.
#include "autodesign.hpp"
#include "params.hpp"
#include "data/field.hpp"

namespace Bakugan {
    class Silhouettes {
    public:
        Silhouettes(const TriMesh& mesh, const AutoParams& p);

        // The loss of a design. With grad: also d loss / d parameter for each of ps, per unit of the parameter.
        AutoScore Loss(const Design& d, const std::vector<Param>& ps, std::vector<float>* grad);

        // The silhouettes scored last against their targets, as an RGBA picture (open pose above, closed below).
        void Picture(const AutoScore& s, std::vector<unsigned char>* rgba, int* w, int* h) const;

        float L() const { return m_L; }
        Vector3 SphereCentre() const { return m_SphereCentre; }
        float SphereRadius() const { return m_SphereRadius; }
        const std::vector<Vector3>& Surface() const { return m_Surface; }   // points on the mesh
    private:
        struct View {
            Vector3 dir, u, v;           // looking along dir; u right and v up in the picture
        };
        // A design's bodies in both poses, and its silhouettes.
        struct Render {
            std::vector<BodyField> fields;
            std::vector<int> group;          // each field's body (group index)
            std::vector<BoundingBox> box;    // design pose
            std::vector<Rigid> xf[2];        // per group index: open, closed
            std::vector<float> cover[2];     // per pose: views × pixels, the union of the bodies
            std::vector<int> who[2];         // ... which field covers it most (-1: none)
            std::vector<Vector3> at[2];      // ... where its ray came closest to that field (world)
        };
        void Draw(const Design& d, Render& r) const;
        float Trace(const BodyField& f, const BoundingBox& box, const Rigid& xf, const View& w, int x, int y, Vector3* at) const;
        Vector2 ToPixel(const View& w, Vector3 p) const;
        float Disc(const View& w, int x, int y) const;   // the sphere's silhouette
    private:
        AutoParams m_P;
        float m_L = 1.0f;
        Vector3 m_Centre = { 0, 0, 0 };  // of the frame every view shows
        float m_Half = 1.0f;             // the frame is 2·half across
        int m_Res = 64;
        float m_Tau = 1.0f;              // half a pixel, in mm: how soft silhouettes are
        std::vector<View> m_Views;
        std::vector<std::vector<float>> m_Mesh;     // the mesh's silhouette per view
        Vector3 m_SphereCentre = { 0, 0, 0 };
        float m_SphereRadius = 0.0f;
        std::vector<Vector3> m_Surface;
        Render m_Last;
    };
}
