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

        // Loads a model file for the reference overlay.
        bool LoadModelFile(const std::string& path, Model* out, std::string* err);

        // Tolerant Wavefront OBJ reader (polygons fan-triangulated, any v/vt/vn index form, negative indices).
        bool ReadOBJ(const std::string& path, TriMesh* out, std::string* err);

        // Copies every mesh of a model into one triangle soup, transformed by xf.
        TriMesh ModelToTriMesh(const Model& m, Matrix xf);

        void ComputeNormals(TriMesh& mesh);

        void ComputeBounds(TriMesh& mesh);

        // Welds coincident vertices, drops degenerate triangles and cancels duplicated faces (pairs facing opposite
        // ways are both removed), so level set meshes come out with clean, two-sided edges.
        void CleanMesh(TriMesh& mesh);

        // Uploads a triangle mesh as a raylib model. Large meshes are split, since raylib indices are 16 bit.
        Model TriMeshToModel(const TriMesh& mesh, Shader lit);

        // Naive surface nets over a dense scalar field (x fastest). Negative values are inside.
        TriMesh SurfaceNets(const std::vector<float>& field, const int dim[3], Vector3 origin, float voxel);

        // Samples f over a box and meshes its zero level set.
        TriMesh MeshImplicit(const std::function<float(Vector3)>& f, BoundingBox box, int resolution);

        Color PartColor(int index);

        Vector3 QuaternionToEulerDegrees(Quaternion q);

        Quaternion EulerDegreesToQuaternion(Vector3 deg);

        Vector3 AnyPerpendicular(Vector3 v);

        // Writes text as is. Formatting stays in memory (Format) and only fopen / fwrite / fclose touch the FILE: on
        // Windows those always come from the same C runtime, while fprintf can come from another one when runtimes are
        // mixed (MinGW msvcrt vs UCRT), which leaves an empty file.
        bool WriteTextFile(const std::string& path, const std::string& text, std::string* err);

        // printf into a std::string.
        std::string Format(const char* fmt, ...);

        // Binary STL, one facet per triangle.
        bool WriteSTL(const std::string& path, const TriMesh& mesh, std::string* err);
    }
}
