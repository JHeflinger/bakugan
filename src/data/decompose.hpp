#pragma once
#include "data/job.hpp"
#include "data/shape.hpp"
#include <memory>
#include <string>
#include <vector>

namespace Bakugan {
    struct DecomposeParams {
        int resolution = 48;          // voxels along the longest side
        int maxParts = 14;
        int rounds = 5;               // residual rounds (seed, fit, prune)
        int seedsPerRound = 4;
        int iterations = 70;          // gradient steps per round
        float parsimony = 0.0006f;    // cost of one more primitive, relative to the reconstruction loss
        float minRegion = 0.006f;     // smallest residual region worth a primitive (fraction of the volume)
        bool refineOnly = false;      // only optimise the given primitives (no seeding or pruning)
        int onlyIndex = -1;           // with refineOnly: optimise just this one, keep the rest fixed
    };

    struct DecomposeResult {
        bool ok = false;
        bool cancelled = false;
        std::string message;
        std::vector<SuperFrustum> shapes;
        float iou = 0.0f;
        double milliseconds = 0.0;
    };

    // Residual primitive fitting in the spirit of SuperFit's ResFit:
    //   1. Morphological shape decomposition seeds primitives in the thickest unexplained regions (erode to the
    //      thickest core, take its connected component, dilate back) and initialises each with PCA.
    //   2. All primitives are refined by gradient descent on a soft occupancy loss with an overlap penalty.
    //   3. Primitives whose removal lowers loss + parsimony are pruned; the unexplained residual seeds the next round.
    // PCA initialisation of one primitive from points of the region it should cover.
    SuperFrustum SeedPrimitiveFromPoints(const std::vector<Vector3>& points, float voxel);

    DecomposeResult Decompose(const std::shared_ptr<const TriMesh>& mesh, const DecomposeParams& params,
                              const std::vector<SuperFrustum>& initial, JobControl& control);
}
