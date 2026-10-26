#pragma once
#include <cstdint>
#include "data/superfrustum.hpp"
#include <string>
#include <vector>

namespace Bakugan {
    // One superfrustum of the design. Lengths are millimetres; the pose is the design pose (how it was modelled).
    struct Shape {
        int id = 0;
        std::string name;
        int group = -1;              // id of the group it belongs to
        SuperFrustum sf;
        Op op = Op::Union;           // Cut removes it from the rest of its group
        Color color = WHITE;
        bool operator==(const Shape& o) const;
    };

    // Groups nest. A top level group (parent -1) is a body: one rigid object that hinges as a whole and prints as one
    // part. A group inside a body is a feature: its shapes (and sub features) are unioned, optionally with a smooth
    // blend, and the result is added to or cut from the enclosing group.
    struct Group {
        int id = 0;
        std::string name;
        int parent = -1;             // id of the enclosing group, -1 = body
        Op op = Op::Union;
        float blend = 0.0f;          // smooth union radius of its members (mm)
        Color color = WHITE;
        bool operator==(const Group& o) const;
    };

    // A revolute joint between two bodies. The child turns about `axis` through `pivot` (both in the design pose,
    // carried by the parent body) by the hinge angle: 0 is the design pose, `start` the open position and `end` the
    // closed one. The whole closing motion runs a fraction t from 0 to 1; this hinge moves while t is in [t0, t1].
    // Opening is the closing motion played backwards unless the hinge has its own opening timing: then it turns while
    // the opening motion (s from 0 = closed to 1 = open) is in [o0, o1], so it can wait until there is room to open.
    // With hardware on it is built as a door (butt) hinge: knuckles alternating between the two bodies around a pin,
    // each set carried by a leaf that fuses into its body, and clearance cut around the other body's knuckles.
    struct Hinge {
        int id = 0;
        std::string name;
        int parent = -1;             // body ids
        int child = -1;
        Vector3 pivot = { 0, 0, 0 };
        Vector3 axis = { 1, 0, 0 };
        float start = 0.0f;          // open angle (radians), -2pi..2pi
        float end = PI * 0.5f;       // closed angle; |end - start| <= 2pi
        float t0 = 0.0f;
        float t1 = 1.0f;
        bool ownOpening = false;
        float o0 = 0.0f;             // share of the opening motion, used with ownOpening
        float o1 = 1.0f;
        bool hardware = true;
        float length = 24.0f;        // along the axis
        float radius = 3.0f;         // knuckle outer radius
        float pinRadius = 1.2f;
        float clearance = 0.4f;      // print gap between the two bodies' moving surfaces
        int knuckles = 3;            // alternating, the parent takes the first, third, ... (the even ones with swapKnuckles)
        bool swapKnuckles = false;   // the child takes the first, third, ... instead
        PinStyle pin = PinStyle::Separate;
        float headDiameter = 0.0f;   // separate pin head; 0 = automatic (the knuckle diameter less the clearance)
        float leafLength = 8.0f;     // from the axis into the body
        float leafThickness = 2.0f;
        bool autoLeaves = true;      // aim each leaf at the centre of its body
        float leafAngle[2] = { 0.0f, 0.0f };   // manual leaf directions (parent, child) about the axis
        uint32_t removed = 0;        // bit k: knuckle k is left out, an open gap on the pin (room for a torsion spring)
        bool operator==(const Hinge& o) const;
    };

    // Knuckle layout along the hinge axis (offsets from the pivot).
    struct Knuckle {
        float lo = 0.0f;
        float hi = 0.0f;
        bool child = false;          // which body carries it
        bool removed = false;        // left out: neither body has material there, both are cleared around it
    };

    // Room left by a run of removed knuckles for a torsion spring around the pin.
    struct SpringSpace {
        int first = 0, last = 0;     // knuckle indices
        float lo = 0.0f, hi = 0.0f;  // along the axis, from the pivot (the cleared span)
        float outer = 0.0f;          // largest coil outer diameter
        float inner = 0.0f;          // smallest coil inner diameter (fits over the pin)
    };

    struct Design {
        std::vector<Shape> shapes;
        std::vector<Group> groups;
        std::vector<Hinge> hinges;
        int nextId = 1;

        int ShapeIndex(int id) const;
        int GroupIndex(int id) const;
        int HingeIndex(int id) const;
        int BodyOf(int groupId) const;                     // id of the top level group containing it, -1 if none
        int ShapeBody(int shapeIndex) const;
        std::vector<int> Bodies() const;                   // indices of the top level groups
        bool Inside(int groupId, int ancestorId) const;    // groupId is ancestorId or nested in it
        std::vector<int> DirectShapes(int groupId) const;  // shape indices
        std::vector<int> ChildGroups(int groupId) const;   // group indices
        std::vector<int> GroupShapes(int groupId) const;   // every shape index in it, recursively
        std::string GroupPath(int groupId) const;          // "Body / Feature"
        int ParentHinge(int bodyId) const;                 // hinge index that turns this body, -1 if none
        bool CanHinge(int parent, int child, int ignoreHinge = -1) const;
        // How far closed (0..1) the hinge is at timeline position t (0 open, 1 closed) of the given motion.
        float HingeFraction(int hinge, float t, bool ease, Motion m = Motion::Closing) const;
        float HingeAngle(int hinge, float t, bool ease, Motion m = Motion::Closing) const;
        bool HasOwnOpening() const;                        // opening is not just closing backwards
        // Body transforms from the design pose to the pose at timeline position t of the motion, indexed like `groups`
        // (features take their body's transform).
        std::vector<Rigid> Transforms(float t, bool ease, Motion m = Motion::Closing) const;
        std::vector<Rigid> PoseTransforms(ExportPose pose, bool ease) const;
        void HingeInWorld(int hinge, const std::vector<Rigid>& xf, Vector3* pivot, Vector3* axis) const;
        Vector3 LeafDirection(int hinge, bool child) const;   // unit, perpendicular to the axis, design pose
        float AutoLeafAngle(int hinge, bool child) const;
        std::vector<Knuckle> Knuckles(int hinge) const;
        // Whether knuckle k can be removed: each body has to keep at least one knuckle on the pin.
        bool KnuckleRemovable(int hinge, int k) const;
        std::vector<SpringSpace> SpringSpaces(int hinge) const;
        float AutoPinHeadDiameter(int hinge) const;
        float MinPinHeadDiameter(int hinge) const;         // just wider than the bore, so the pin can't slide through
        // Head of a separate pin, past the last knuckle: axial range from the pivot and radius.
        void PinHead(int hinge, float* from, float* to, float* radius) const;
        bool AddsMaterial(int shapeIndex) const;           // unioned, and not inside a cut feature
        BoundingBox BodyBounds(int bodyId) const;          // design pose, added material only
        BoundingBox Bounds() const;

        int AddGroup(const std::string& name, int parent);
        int AddShape(const SuperFrustum& sf, int groupId, const std::string& name = "", Op op = Op::Union);
        int AddHinge(int parent, int child);
        void RemoveShape(int index);
        void RemoveGroup(int index);                       // with everything in it and its hinges
        void RemoveHinge(int index);
        bool MoveShape(int index, int groupId);
        bool MoveGroup(int index, int parentId);           // -1 makes it a body
        int GroupItems(const std::vector<int>& shapeIds, const std::vector<int>& groupIds);   // new group id
        void Ungroup(int index);
        int DuplicateShape(int index, Vector3 offset);
        int DuplicateGroup(int index, int parentId, Vector3 offset);
        void Repair();
        std::string UniqueName(const std::string& base) const;
        Color NextColor() const;

        std::string Serialize() const;
        bool Deserialize(const std::string& text, std::string* err);
        bool operator==(const Design& o) const;
        bool operator!=(const Design& o) const { return !(*this == o); }
    };

    // Starting points for the template list.
    std::vector<std::string> TemplateNames();
    Design MakeTemplate(int index);
}
