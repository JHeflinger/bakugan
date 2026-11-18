#pragma once
// What the generator designs: a sphere carved into pieces, hinged together.
//
// A carving is a sphere and a set of cutters (superfrustums). Every cutter splits one piece in two: the part inside the
// cutter and the part outside it. So a piece is the sphere, inside some cutters, outside others; the pieces fill the
// sphere exactly, and no two of them share any material. Pieces are joined by joints (hinges): a joint turns its child
// piece about an axis through a pivot, from the closed angle 0 (where every piece sits in the sphere) to an open angle.
// So the closed pose is the sphere, whatever the parameters; the open pose is what the loss compares with the mesh.
//
// A carving becomes an ordinary Design (Build) for the loss, the editor and export: a body per piece, holding the sphere,
// a cut for every cutter it's outside of, and a cut feature "outside this cutter" (a box around the
// sphere, less the cutter) for every cutter it's inside of; a hinge per joint (closed angle 0).
//
// Its differentiable parameters (Adam's): every cutter's superfrustum numbers, every joint's pivot, axis, open angle and
// timing, and the placement of the mesh it's compared with (an offset and a scale).
#include "data/design.hpp"
#include <cstdint>
#include <vector>

namespace Bakugan {
    struct Cutter {
        int id = 0;
        SuperFrustum sf;
        bool core = false;                       // the cutter that splits the sphere into a shell and a core: no rule removes it
    };

    // A part of the sphere: inside some cutters, outside others.
    struct Region {
        std::vector<int> inside, outside;        // cutter ids
    };

    // A piece: one region, and any others joined to it (a panel of the shell thickened with part of the core: "thicken
    // a panel"). A cutter that splits a piece splits every region of it.
    struct Piece {
        int id = 0;
        std::vector<int> inside, outside;        // its first region's cutter ids
        std::vector<Region> extra;               // the regions joined to it
    };

    struct Joint {
        int id = 0;
        int parent = -1, child = -1;             // piece ids
        Vector3 pivot = { 0, 0, 0 };             // in the closed pose (the sphere's frame)
        Vector3 axis = { 1, 0, 0 };
        float open = 0.0f;                       // the angle it opens to (radians); closed is 0
        float t0 = 0.0f, t1 = 1.0f;              // when it turns, as the design closes (0..1)
        int order = 0;                           // its place in the opening order (0 opens first), when the carving is ordered
    };

    struct Carving {
        Vector3 centre = { 0, 0, 0 };
        float radius = 1.0f;
        std::vector<Cutter> cutters;
        std::vector<Piece> pieces;
        std::vector<Joint> joints;
        int nextId = 1;
        // Ordered: the joints open one after another, each in its own share of the motion, in the order of their
        // `order` (Tidy sets t0 and t1 from it: timing is then no parameter). Otherwise t0 and t1 are Adam's.
        bool ordered = false;
        // The core stays shut inside the shell (the first stage of a search that starts from a shell and a core):
        // nothing turns it or cuts it, and its cutter and hinge are no parameters, so only the shell's pieces change.
        bool coreLocked = false;

        // One piece: the whole sphere.
        static Carving Sphere(Vector3 centre, float radius);
        // Two pieces: a shell, and a core inside it (a ball `core` across, at the centre), hinged
        // to the shell, closed. The shell's pieces make thin parts (wings, a tail), the core's chunky ones (a head, a
        // body): it comes out once the shell's pieces have swung out of its way.
        static Carving ShellAndCore(Vector3 centre, float radius, float core);

        // As a design: a body per piece (in pieces' order: bodyOf[k] is piece k's body group id), a hinge per joint.
        Design Build(std::vector<int>* bodyOf = nullptr) const;

        int PieceIndex(int id) const;
        int CutterIndex(int id) const;
        int JointIndex(int id) const;
        // The pieces whose shape depends on a cutter (as indices).
        std::vector<int> PiecesUsing(int cutterId) const;
        // Splits piece pi (index) with a cutter already in `cutters`: the new piece (its index, last) is the part of
        // every region inside the cutter, the piece keeps the part outside it.
        int Split(int pi, int cutterId);
        // Whether a piece is (all or partly) the core's: a region of it inside the core's cutter.
        bool InCore(const Piece& p) const;
        // Whether a joint turns the core, or a piece of it, against the shell (or within it).
        bool CoreJoint(const Joint& j) const;
        // A box around a piece: the sphere's, cut down by the cutters it's inside of.
        BoundingBox PieceBounds(const Piece& p) const;
        // The joint a piece hangs from (its index, or -1 for the root).
        int ParentJoint(int pieceId) const;
        // The joints' ids in opening order.
        std::vector<int> Order() const;
        // Sets the opening order (joint ids, first opens first; joints left out keep their order after those).
        void SetOrder(const std::vector<int>& ids);
    };

    // Where the mesh is placed for the comparison: scaled about its own centre (of volume), then moved.
    struct Placement {
        Vector3 offset = { 0, 0, 0 };
        float scale = 1.0f;
        Vector3 centre = { 0, 0, 0 };            // the mesh's centre of volume (not a parameter: the loss fills it in)
        Vector3 Apply(Vector3 p) const {
            return { centre.x + offset.x + scale * (p.x - centre.x), centre.y + offset.y + scale * (p.y - centre.y),
                     centre.z + offset.z + scale * (p.z - centre.z) };
        }
    };

    // A differentiable number: a cutter's, a joint's, or the mesh's placement's.
    struct Param {
        enum Kind : unsigned char { Cutter, Joint, Target } kind;
        int id;                  // cutter or joint id (0 for the target)
        int field;               // which of its numbers (a row of its table in carving.cpp)
        float unit;              // a typical step, in the number's own units (mm, radians, ...): Adam steps in units
        uint64_t Key() const { return ((uint64_t)kind << 56) | ((uint64_t)(uint32_t)id << 8) | (uint64_t)field; }
    };

    // The carving's parameters (cutters', then joints'), in a fixed order. L: the mesh's size (lengths scale with it).
    std::vector<Param> CarvingParams(const Carving& c, float L);
    // The placement's: offset (3, if moving) and log scale (1, if scaling).
    std::vector<Param> TargetParams(bool move, bool scale, float L);
    // A parameter's value (nullptr when its cutter or joint is gone); the target's log scale is stored as the scale.
    float Get(const Carving& c, const Placement& t, const Param& p, bool* ok = nullptr);
    // Sets it, kept inside its range (sizes stay positive, rounding within 0..1, ...).
    void Set(Carving& c, Placement& t, const Param& p, float value, float L);
    // After changing parameters: rotations and axes back to unit length, joints' timing windows kept in order (or, in an
    // ordered carving, set from the opening order: equal shares of the motion, the first to open the last to close).
    void Tidy(Carving& c);
}
