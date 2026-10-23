#pragma once
#include "data/build.hpp"
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace Bakugan {
    // One thing that runs into another while the design closes: a shape (or hinge hardware) of one body digging
    // into a shape (or hardware) of another.
    struct Collision {
        int bodyA = -1, bodyB = -1;      // group ids
        int shapeA = -1, shapeB = -1;    // shape ids (-1 when it is hinge hardware)
        int hingeA = -1, hingeB = -1;    // hinge ids of the hardware
        float firstT = 1.0f, lastT = 0.0f;
        float worstT = 0.0f;
        float depth = 0.0f;              // deepest penetration (mm)
        Vector3 point = { 0, 0, 0 };     // world position at worstT
        Vector3 local = { 0, 0, 0 };     // the same spot in the design pose (where it is in the design view)
        int steps = 0;                   // checked moments it shows up in
    };

    // What collides at one checked moment of the motion.
    struct SweepStep {
        float t = 0.0f;
        float depth = 0.0f;              // 0 = clear
        std::vector<Vector3> points;     // world positions (a capped sample)
        std::vector<int> shapes;         // shape ids involved
        std::vector<int> bodies;         // body ids involved
        std::vector<int> hinges;         // hinge ids whose hardware is involved
    };

    struct SweepParams {
        int steps = 72;
        float tolerance = 0.05f;         // penetrations shallower than this are touching, not colliding
        bool ease = true;
        bool operator==(const SweepParams& o) const { return steps == o.steps && tolerance == o.tolerance && ease == o.ease; }
        bool operator!=(const SweepParams& o) const { return !(*this == o); }
    };

    // A surface sample of one body found inside another, in the first body's design frame.
    struct SweepHit {
        Vector3 local = { 0, 0, 0 };
        float depth = 0.0f;
        bool named = false;              // attributed to the shapes below (only a sample of the hits is)
        int shapeA = -1, hingeA = -1;
        int shapeB = -1, hingeB = -1;
    };

    // Hits of an ordered pair of bodies, per checked moment, kept between sweeps: a pair whose two parts and whose
    // relative motion are unchanged is not checked again.
    struct SweepCache {
        std::mutex mutex;
        std::unordered_map<std::string, std::shared_ptr<const std::vector<std::vector<SweepHit>>>> pairs;
    };

    struct SweepResult {
        bool ok = false;
        bool cancelled = false;
        std::string message;
        Design design;
        SweepParams params;
        std::vector<SweepStep> steps;    // t = k / (steps - 1)
        std::vector<Collision> collisions;   // deepest first
        int pairsChecked = 0;
        int pairsReused = 0;
        bool Clear() const { return collisions.empty(); }
        int StepAt(float t) const;
    };

    // Moves the design from open (t = 0) to closed (t = 1) and checks every body's surface samples against every
    // other body's signed distance field at each step.
    SweepResult SweepCollisions(const Design& d, std::shared_ptr<const std::vector<BuiltPart>> parts, const SweepParams& p,
                                JobControl& c, std::shared_ptr<SweepCache> cache = nullptr);
}
