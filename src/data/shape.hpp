#pragma once
#include "util/structs.hpp"
#include <memory>
#include <string>
#include <vector>

namespace Bakugan {
    // A dense signed distance field (negative inside) sampled at voxel centres, x fastest.
    struct ShapeField {
        Vector3 origin = { 0, 0, 0 };     // centre of voxel (0,0,0)
        float voxel = 0.0f;
        int dim[3] = { 0, 0, 0 };
        std::vector<float> sdf;

        bool Valid() const { return !sdf.empty(); }
        size_t Index(int i, int j, int k) const { return ((size_t)k * dim[1] + j) * dim[0] + i; }
        Vector3 Position(int i, int j, int k) const { return { origin.x + i * voxel, origin.y + j * voxel, origin.z + k * voxel }; }
        float Sample(Vector3 p) const;    // trilinear; positive (outside) beyond the grid
        float Laplacian(Vector3 p) const;
    };

    struct DetailSample {
        Vector3 position;
        float weight = 1.0f;              // higher on creases and small features
    };

    // Everything the optimisers need to know about the input shape.
    struct ShapeAnalysis {
        bool ok = false;
        std::string message;
        ShapeField field;
        double volume = 0.0;
        float equalRadius = 0.0f;         // radius of the sphere with the same volume
        Vector3 centroid = { 0, 0, 0 };
        std::vector<Vector3> interior;    // centres of the interior voxels of `field`
        std::vector<DetailSample> detail; // points on the surface
    };

    // Level set of a closed triangle mesh (OpenVDB mesh to volume), stored densely with the interior filled in.
    ShapeField BuildShapeField(const TriMesh& mesh, float voxel, float padding, std::string* err);

    ShapeAnalysis AnalyzeShape(const TriMesh& mesh, int resolution, int detailSamples);
}
