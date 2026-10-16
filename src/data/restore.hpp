#pragma once
#include "data/job.hpp"
#include "data/shape.hpp"
#include <memory>
#include <string>
#include <vector>

namespace Bakugan {
    struct RestoreParams {
        int resolution = 80;            // voxels across the sphere's diameter
        float shellThickness = 0.12f;   // wall of the outer sphere shell, relative to the radius
        float gapFill = 0.10f;          // shell plates may reach this far (relative to the radius) past their superfrustum
        bool fillGaps = true;
        float adaptivity = 0.0f;        // OpenVDB mesh simplification, 0..1
    };

    struct RestoredPart {
        int partId = 0;
        TriMesh mesh;                   // open pose (figure space)
        float volume = 0.0f;
    };

    struct RestoreResult {
        bool ok = false;
        bool cancelled = false;
        std::string message;
        Design design;                  // the design the parts were built for
        std::vector<RestoredPart> parts;
        float shellCoverage = 0.0f;
        double milliseconds = 0.0;
    };

    // Builds the final printable pieces. Each part is
    //     ((input material in the part's cell)  union  (its share of the sphere shell))  intersect  (the sphere)
    // evaluated in the closed pose, minus the pieces of higher priority parts it would collide with. The superfrustums
    // act as the bounding boxes of their features: they decide the cells and the shell plates, while every surface that
    // is not on the outer sphere comes straight from the input's level set, so its detail is restored.
    RestoreResult RestoreDetail(const Design& design, const std::shared_ptr<const TriMesh>& mesh, const RestoreParams& params,
                                JobControl& control);
}
