#pragma once
// Overlaps between bodies on their exact fields, and where in the motion to look for them. Shared by the generator's
// collision term (autodesign/loss.cpp) and the finishing passes (data/finish.cpp).
//
// Where: every hinge's turn gets the same number of checks, spread evenly over its angle, however long or short the
// turn is on the timeline. A hinge that turns in a twentieth of the motion is looked at as closely as one that takes
// all of it, so nothing is gained (or hidden) by turning fast.
//
// How much: two bodies are sampled on a grid over the box where they can meet. A cell the fields show to be clear of
// either body is skipped, one that is inside both counts whole, and one their surfaces pass through is split in eight
// and sampled again. The fields bound the distance, so an overlap thicker than half a cell can't fall between samples.
#include "data/design.hpp"
#include "data/field.hpp"
#include <vector>

namespace Bakugan {
    // A moment of the closing motion that is checked: when (0 open .. 1 closed), and the hinge whose turn it samples.
    struct MotionCheck {
        float t = 0.0f;
        int hinge = -1;              // index
        float fraction = 0.0f;       // of that hinge's angle, 0 open .. 1 closed
    };

    // perHinge checks in every moving hinge's turn, at fractions (k + phase) / perHinge of its angle (phase in 0..1: a
    // different one each time looks between the last ones). With ease, the times are those the eased motion
    // (Design::Transforms(t, true)) reaches those angles at.
    std::vector<MotionCheck> HingeChecks(const Design& d, int perHinge, float phase, bool ease = true);

    // under[h][g]: body (group index) g turns with hinge h (it is the hinge's child, or hangs from it).
    std::vector<std::vector<char>> HingeSubtrees(const Design& d);
    // Whether two bodies (group indices) move against each other at time t: a hinge that is turning then has one of
    // them on its child's side and the other not.
    bool MovingApart(const Design& d, const std::vector<std::vector<char>>& under, int ga, int gb, float t);
    // Hinge indices in the order they open: the last to close first (by the end of their window; ties by index).
    std::vector<int> OpeningOrder(const Design& d);

    // A body placed in the world.
    struct Solid {
        const BodyField* field = nullptr;
        BoundingBox box = {};        // design pose: nothing of the body is outside it
        Rigid xf;
    };

    struct OverlapSample {
        Vector3 at = { 0, 0, 0 };    // world
        float depth = 0.0f;          // how deep inside both it is (the lesser of its two depths), > 0
        float volume = 0.0f;         // the part of space it stands for
        bool first = false;          // the lesser depth is the first solid's (its surface is the nearer one)
    };

    struct Clash {
        double volume = 0.0;         // inside both
        double depth = 0.0;          // depth integrated over that volume
        float deepest = 0.0f;
        std::vector<OverlapSample> samples;
    };

    // Adds what two solids share to `out`. cell: the grid's spacing (cells on the surfaces are sampled at half of it);
    // jitter (each in 0..1) shifts the grid. Samples are kept only when asked for.
    void SampleOverlap(const Solid& a, const Solid& b, float cell, Vector3 jitter, Clash* out, bool keepSamples);

    // The fields and boxes of a design's bodies (those that have any shape), in Bodies() order.
    struct BodySet {
        std::vector<BodyField> fields;
        std::vector<BoundingBox> boxes;
        std::vector<int> group;      // each one's group index
        std::vector<int> fieldOf;    // per group index: its field, or -1
    };
    BodySet MakeBodySet(const Design& d, float grow);

    // The whole motion checked: the open pose, and perHinge poses in every hinge's turn.
    struct MotionOverlap {
        double openVolume = 0.0;     // mm^3 inside two bodies at once, open
        double worstVolume = 0.0;    // ... at the worst checked pose between
        float worstT = 0.0f;
        int worstHinge = -1;         // index
        float deepest = 0.0f;        // mm, anywhere
        int poses = 0;
        std::vector<double> perHinge;    // the worst volume in each hinge's turn (by hinge index)
    };
    MotionOverlap MeasureMotion(const Design& d, int perHinge, float cell, float phase = 0.5f);
}
