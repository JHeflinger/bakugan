#pragma once
#include <raylib.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace Bakugan {
    struct MeshData {
        uint64_t id = 0;
        std::vector<Vector3> vertices;
        std::vector<unsigned> triangles;
        BoundingBox box = {};
        static uint64_t NextId();
    };
    struct OverlapParams {
        int resolution = 64;
        float erosion = 0.5f;
    };

    struct OverlapRequest {
        std::shared_ptr<const MeshData> source;   // the left mesh, rest pose, left world space
        std::shared_ptr<const MeshData> target;   // the right mesh, right world space
        std::vector<Matrix> restWorld;            // joint world transforms of the left rig
        std::vector<Matrix> posedWorld;           // joint world transforms of the right (posed) rig
        std::vector<int> parent;                  // joint parent indices (shared by both rigs)
        Matrix place = {};                        // left world space -> right world space for the rig
        OverlapParams params;
        uint64_t signature = 0;                   // identifies the inputs, echoed back in the result
    };

    // Voxel grid over the right mesh and the folded left mesh. Cells: 0 = nothing, 1 = folded mesh inside the right
    // mesh (fine), 2 = folded parts overlap inside the right mesh, 3 = empty space inside the right mesh,
    // 4 = folded mesh outside the right mesh.
    struct OverlapResult {
        bool ok = false;
        std::string message;                // human readable status or error
        uint64_t signature = 0;
        Vector3 origin = { 0, 0, 0 };       // world position of the centre of voxel (0,0,0)
        float voxel = 0.0f;
        int dim[3] = { 0, 0, 0 };
        std::vector<uint8_t> cells;         // x fastest, then y, then z
        // The same classification mapped back onto the left mesh (rest pose, left world space):
        // 0 = outside the mesh, 1 = inside and fine, 2 = inside and its folded position overlaps.
        Vector3 sourceOrigin = { 0, 0, 0 };
        float sourceVoxel = 0.0f;
        int sourceDim[3] = { 0, 0, 0 };
        std::vector<uint8_t> sourceCells;
        double targetVolume = 0.0;          // volume of the right mesh (world units^3)
        double overlapVolume = 0.0;         // volume of the folded mesh that overlaps itself or sticks out
        double milliseconds = 0.0;
    };

    // Folds the left mesh into the right mesh's space using the posed skeleton and finds where
    // different bones' parts land on top of each other. The work is done on OpenVDB level sets:
    //   * a level set of the right mesh decides which voxels are inside it,
    //   * a level set of the left mesh (rest pose) decides which rest-space points are solid.
    // Each bone owns the part of the left mesh closest to it. A voxel of the right mesh is "overlapping"
    // when two or more bones' rigidly moved parts both cover it.
    // Runs on a worker thread; only one job is in flight, and a newer request replaces a waiting one.
    class OverlapField {
    public:
        OverlapField();
        ~OverlapField();
        OverlapField(const OverlapField&) = delete;
        OverlapField& operator=(const OverlapField&) = delete;

        void Request(OverlapRequest request);
        bool Poll(OverlapResult* out);      // true when a finished result was moved into *out
        bool Busy() const;

        // Blocking version, used by tests.
        static OverlapResult Compute(const OverlapRequest& request);
    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };
}
