#pragma once
// The continuous half of the design space: every number the descent may move, with the structure fixed.
#include "space.hpp"

namespace Bakugan::Auto {
    // One kind of continuous parameter. To add one: a row in the table in params.cpp (its part and unit), where it is
    // listed in Parameters::Of, and how a step moves it in Parameters::Step.
    enum class Dof : unsigned char {
        Pos, LogSize, Rot, Rounding, Taper, Bulge, Dilation,     // a shape (sizes on a log scale, rotation as a small turn)
        Pivot, Axis, Angle, Timing,                              // a hinge
        Blend                                                    // a group of two or more shapes
    };

    // Parameters moved together in one descent step (in few dimensions a random probe direction lines up with the
    // gradient far more often).
    enum class Part : signed char { All = -1, Position, Size, Rotation, Profile, Pivot, Axis, Angle, Timing, Blend };

    struct Param {
        Dof dof;
        int id;                      // shape, hinge or group id
        int axis;                    // 0..2 for the vector ones (Timing: 0 slide, 1 stretch the closing window; 2, 3
                                     // the same for the opening window)
    };

    // A part of a design: one element (a shape, a hinge or a group), and one part of its parameters or all of them.
    // Also what a rewrite touched, so its lookahead tunes only that.
    struct Focus {
        int shape = -1;              // shape id
        int shape2 = -1;             // a second shape (the other half of a split)
        int hinge = -1;              // hinge id
        int group = -1;              // group id (its blend)
        Part part = Part::All;
    };

    class Parameters {
    public:
        explicit Parameters(const DesignSpace& s) : m_Space(s) {}
        // One unit of a kind: a small change comparable to one unit of any other (the paper's grammar-specific
        // scaling), relative to the mesh's size where it is a length.
        float Unit(Dof k) const;
        Part PartOf(Dof k) const;
        // The parameters of a design, or of the focused element and part only.
        std::vector<Param> Of(const Design& d, const Focus* only = nullptr) const;
        // Moves every parameter by x[i] units, then pulls the design back into the design space.
        void Step(Design& d, const std::vector<Param>& ps, const std::vector<float>& x) const;
        // What the next descent step moves: a shape, a hinge or a blended group, each as likely as the next, and
        // usually one part of its parameters, now and then (one in five) all of them.
        Focus RandomBlock(const Design& d, Rng& rng) const;
    private:
        const DesignSpace& m_Space;
    };
}
