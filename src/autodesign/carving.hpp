#pragma once
// What the generator designs: a sphere carved into pieces, hinged together.
//
// A carving is a sphere and a set of cutters (superfrustums). Every cutter splits one piece in two: the part inside the
// cutter and the part outside it. So a piece is the sphere, inside some cutters, outside others; the pieces fill the
// sphere exactly, and no two of them share any material. Hollows are cutters too, that only take material away: from
// every piece, without making one. Then the pieces fill the sphere less the hollows (the loss's surface term keeps
// hollows inside, so the sphere's surface stays whole). Pieces are joined by joints (hinges): a joint turns its child
// piece about an axis through a pivot, from the closed angle 0 (where every piece sits in the sphere) to an open angle.
// So the closed pose is the sphere, whatever the parameters; the open pose is what the loss compares with the mesh.
//
// A carving becomes an ordinary Design (Build) for the loss, the editor and export: a body per piece, holding the sphere,
// a cut for every cutter it's outside of and every hollow, and a cut feature "outside this cutter" (a box around the
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
        bool hollow = false;                     // takes material away from every piece, instead of splitting one
    };

    struct Piece {
        int id = 0;
        std::vector<int> inside, outside;        // cutter ids
    };

    struct Joint {
        int id = 0;
        int parent = -1, child = -1;             // piece ids
        Vector3 pivot = { 0, 0, 0 };             // in the closed pose (the sphere's frame)
        Vector3 axis = { 1, 0, 0 };
        float open = 0.0f;                       // the angle it opens to (radians); closed is 0
        float t0 = 0.0f, t1 = 1.0f;              // when it turns, as the design closes (0..1)
    };

    struct Carving {
        Vector3 centre = { 0, 0, 0 };
        float radius = 1.0f;
        std::vector<Cutter> cutters;
        std::vector<Piece> pieces;
        std::vector<Joint> joints;
        int nextId = 1;

        // One piece: the whole sphere.
        static Carving Sphere(Vector3 centre, float radius);

        // As a design: a body per piece (in pieces' order: bodyOf[k] is piece k's body group id), a hinge per joint.
        Design Build(std::vector<int>* bodyOf = nullptr) const;

        int PieceIndex(int id) const;
        int CutterIndex(int id) const;
        int JointIndex(int id) const;
        // The pieces whose shape depends on a cutter (as indices): a hollow's, all of them.
        std::vector<int> PiecesUsing(int cutterId) const;
        int Hollows() const;
        // A box around a piece: the sphere's, cut down by the cutters it's inside of.
        BoundingBox PieceBounds(const Piece& p) const;
        // The joint a piece hangs from (its index, or -1 for the root).
        int ParentJoint(int pieceId) const;
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
    // After changing parameters: rotations and axes back to unit length, joints' timing windows kept in order.
    void Tidy(Carving& c);
}
