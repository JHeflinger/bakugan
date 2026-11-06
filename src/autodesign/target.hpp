#pragma once
// The input mesh, measured once before the search: what every design is compared against.
#include "common.hpp"
#include "data/job.hpp"
#include <memory>

namespace Bakugan::Auto {
    // Inside or outside the mesh: ray crossings along +x, +y and +z, the majority of the three parities, so a mesh
    // with a few holes or doubled faces still mostly answers right. Triangles are binned on a grid per axis.
    class InsideTest {
    public:
        InsideTest(const TriMesh& m, BoundingBox box);
        bool Inside(Vector3 p) const;
    private:
        struct Axis {
            int a = 0, u = 1, v = 2, n = 1;
            float u0 = 0, v0 = 0, du = 1, dv = 1;
            std::vector<std::vector<unsigned>> cells;
        };
        int Crossings(const Axis& ax, Vector3 p) const;
        const TriMesh& m_Mesh;
        Axis m_Axes[3];
    };

    // The mesh's signed distance (negative inside) on a grid around it, trilinear in between and growing with the
    // distance to the grid outside it: how far design material is from the mesh, wherever it is.
    class MeshField {
    public:
        void Build(BoundingBox box, float pad, int res, const InsideTest& inside, Pool& pool);
        float Distance(Vector3 p) const;
        double Volume() const { return m_Volume; }
    private:
        Vector3 Point(int x, int y, int z) const { return { m_Min.x + x * m_H, m_Min.y + y * m_H, m_Min.z + z * m_H }; }
        Vector3 m_Min = { 0, 0, 0 };
        float m_H = 1.0f;
        int m_N[3] = { 2, 2, 2 };
        std::vector<float> m_D;
        double m_Volume = 0.0;
    };

    struct Target {
        const TriMesh* mesh = nullptr;
        float L = 1.0f;                  // diagonal of the mesh's box: the length scale of everything
        BoundingBox box = {};
        Vector3 centre = { 0, 0, 0 };    // of the mesh's volume
        std::vector<Vector3> vol;        // volume samples (uniform in the box and near the surface) ...
        std::vector<char> volIn;         // ... and whether each is inside the mesh
        std::vector<Vector3> surf;       // on the mesh surface (area weighted)
        std::vector<Vector3> inside;     // a pool of points inside the mesh (seeding, placing new shapes)
        std::unique_ptr<InsideTest> test;
        MeshField field;

        // Samples the mesh at the given quality (0..2). False when it has no usable inside, or when cancelled.
        bool Build(const TriMesh& m, int quality, Rng& rng, Pool& pool, JobControl& c);
        bool Inside(Vector3 p) const { return test->Inside(p); }
    };
}
