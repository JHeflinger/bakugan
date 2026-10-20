#pragma once
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
        bool hardware = true;
        float length = 24.0f;        // along the axis
        float radius = 3.0f;         // knuckle outer radius
        float pinRadius = 1.2f;
        float clearance = 0.4f;      // print gap between the two bodies' moving surfaces
        int knuckles = 3;            // alternating, the parent takes the first and last
        PinStyle pin = PinStyle::Separate;
        float leafLength = 8.0f;     // from the axis into the body
        float leafThickness = 2.0f;
        bool autoLeaves = true;      // aim each leaf at the centre of its body
        float leafAngle[2] = { 0.0f, 0.0f };   // manual leaf directions (parent, child) about the axis
        bool operator==(const Hinge& o) const;
    };

    // Knuckle layout along the hinge axis (offsets from the pivot).
    struct Knuckle {
        float lo = 0.0f;
        float hi = 0.0f;
        bool child = false;          // which body carries it
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
        float HingeFraction(int hinge, float t, bool ease) const;
        float HingeAngle(int hinge, float t, bool ease) const;
        // Body transforms from the design pose to the pose at fraction t, indexed like `groups` (features take their
        // body's transform).
        std::vector<Rigid> Transforms(float t, bool ease) const;
        std::vector<Rigid> PoseTransforms(ExportPose pose, bool ease) const;
        void HingeInWorld(int hinge, const std::vector<Rigid>& xf, Vector3* pivot, Vector3* axis) const;
        Vector3 LeafDirection(int hinge, bool child) const;   // unit, perpendicular to the axis, design pose
        float AutoLeafAngle(int hinge, bool child) const;
        std::vector<Knuckle> Knuckles(int hinge) const;
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
