#pragma once
#include "data/job.hpp"
#include "data/field.hpp"
#include <memory>
#include <string>
#include <vector>

namespace Bakugan {
    // One printable part, meshed in the design pose.
    struct BuiltPart {
        int body = -1;               // group id, or -1 for a separate pin
        int pinHinge = -1;           // hinge id of a separate pin
        std::string name;
        TriMesh mesh;
        std::vector<Vector3> samples;   // points on the true surface, for collision checks
    };

    struct BuildResult {
        bool ok = false;
        bool cancelled = false;
        std::string message;
        Design design;               // what was built
        float voxel = 1.0f;
        std::vector<BuiltPart> parts;
    };

    // Meshes the zero level set of f inside box (OpenVDB narrow band level set, evaluated sparsely: only blocks the
    // field says the surface can pass through are sampled densely). progress receives 0..1.
    TriMesh MeshLevelSet(const std::function<float(Vector3)>& f, BoundingBox box, float voxel, float adaptivity,
                         const JobControl* control, const std::function<void(float)>& progress);

    // Meshes every body (and separate pin) of the design. samples > 0 also takes up to that many surface points per part.
    // With a pose (transforms indexed like the groups) and gap > 0, every body is trimmed back by gap / 2 from the bodies
    // it touches in that pose, so parts printed together come out apart.
    BuildResult BuildParts(const Design& d, float voxel, float adaptivity, int samples, JobControl& c,
                           const std::vector<Rigid>* pose = nullptr, float gap = 0.0f);

    // Preview voxel size for a design: its longest side over `resolution`.
    float PreviewVoxel(const Design& d, int resolution);

    struct ExportParams {
        std::string prefix = "bakugan_parts";
        float voxel = 0.2f;          // mm
        float adaptivity = 0.1f;     // OpenVDB mesh simplification
        ExportPose pose = ExportPose::Open;
        bool ease = true;
        bool separate = true;        // one STL per part
        bool combined = false;       // one STL with every part (print in place)
        bool keepApart = true;       // trim bodies that touch in the export pose
        float gap = 0.4f;            // ... to this gap (mm)
        bool zUp = true;             // slicers use +Z up
        bool dropToBed = true;       // lowest point at z = 0 (whole assembly)
        float scale = 1.0f;          // millimetres per design unit
    };

    struct ExportResult {
        bool ok = false;
        bool cancelled = false;
        std::string message;
        std::vector<std::string> files;
    };

    ExportResult ExportSTL(const Design& d, const ExportParams& p, JobControl& c);
}
