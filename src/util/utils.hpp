#pragma once
#include "util/structs.hpp"
#include <functional>
#include <string>
#include <vector>

namespace Bakugan {
    namespace Utils {
        const char* GetVertexShader();

        const char* GetFragmentShader();

        bool IsModelFile(const char* path);

        bool LoadAsset(const Asset& a, Model* out, std::string* err);

        // Tolerant Wavefront OBJ reader (polygons fan-triangulated, any v/vt/vn index form, negative indices).
        bool ReadOBJ(const std::string& path, TriMesh* out, std::string* err);

        // Copies every mesh of a model into one triangle soup, transformed by xf.
        TriMesh ModelToTriMesh(const Model& m, Matrix xf);

        void ComputeNormals(TriMesh& mesh);

        void ComputeBounds(TriMesh& mesh);

        // Uploads a triangle mesh as a raylib model. Large meshes are split, since raylib indices are 16 bit.
        Model TriMeshToModel(const TriMesh& mesh, Shader lit);

        // Naive surface nets over a dense scalar field (x fastest). Negative values are inside.
        TriMesh SurfaceNets(const std::vector<float>& field, const int dim[3], Vector3 origin, float voxel);

        // Samples f over a box and meshes its zero level set.
        TriMesh MeshImplicit(const std::function<float(Vector3)>& f, BoundingBox box, int resolution);

        std::vector<Vector3> FibonacciSphere(int count);

        Color PartColor(int index);

        Vector3 QuaternionToEulerDegrees(Quaternion q);

        Quaternion EulerDegreesToQuaternion(Vector3 deg);

        Vector3 AnyPerpendicular(Vector3 v);

        bool WriteOBJ(const std::string& path, const std::vector<const TriMesh*>& meshes, const std::vector<Rigid>& transforms,
                      const std::vector<std::string>& names, std::string* err);
    }
}
