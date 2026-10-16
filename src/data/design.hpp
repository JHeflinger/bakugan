#pragma once
#include "data/superfrustum.hpp"
#include <string>
#include <vector>

namespace Bakugan {
    // A spring loaded, uni-directional revolute hinge joining a part to its parent. In the open pose (the figure) the
    // spring holds the hinge at 0. Folding rotates the part about `axis` through `pivot` by `angle` (only positive
    // rotation is possible, up to maxAngle). Pivot and axis are given in the open pose and move with the parent.
    struct Hinge {
        int parent = -1;
        Vector3 pivot = { 0, 0, 0 };
        Vector3 axis = { 1, 0, 0 };
        float angle = 0.0f;          // closed (ball) angle, radians, 0..maxAngle
        float maxAngle = PI;
        bool locked = false;         // rigid connection instead of a hinge
        // Release chain (single latch): the master hinge (trigger -1) is held by the latch on the root; every other
        // working hinge is held shut by a catch on the part of its trigger hinge, and springs open once that hinge
        // has opened `release` of its travel. -2 = not assigned yet (repaired to the default chain).
        int trigger = -2;
        float release = 0.5f;
        bool operator==(const Hinge& o) const;
    };

    struct Part {
        int id = 0;
        std::string name;
        SuperFrustum shape;          // in the open pose (world space of the figure)
        Hinge hinge;
        Color color = WHITE;
        bool operator==(const Part& o) const;
    };

    struct Ball {
        Vector3 center = { 0, 1, 0 };
        float radius = 1.0f;
        bool operator==(const Ball& o) const;
    };

    // When each hinge starts and finishes opening, in units where the whole pop-open takes `total`.
    struct ReleasePlan {
        std::vector<float> start;
        std::vector<float> duration;
        float total = 1.0f;
    };

    // The full folding design: superfrustum parts arranged in a hinge tree, and the target sphere.
    struct Design {
        std::vector<Part> parts;
        Ball ball;
        int nextId = 1;
        bool chained = true;         // hinges open one after another from a single latch (else all at once)

        int Root() const;
        int IndexOf(int id) const;
        // Part transforms from the open pose to the pose at fold fraction t (0 = figure, 1 = ball).
        std::vector<Rigid> Transforms(float t) const;
        void HingeInWorld(int part, const std::vector<Rigid>& xf, Vector3* pivot, Vector3* axis) const;
        float FoldAngle(int part) const;     // effective closed angle (0 when locked / unattached)
        std::vector<int> Children(int part) const;
        std::vector<int> Subtree(int part) const;          // part and all its descendants
        bool IsDescendant(int part, int ancestor) const;
        std::vector<int> TopologicalOrder() const;         // parents before children
        int Depth(int part) const;
        int AddPart(const SuperFrustum& shape, int parent, const std::string& name = "");
        void RemovePart(int part);                         // children move to the removed part's parent
        bool Reparent(int part, int parent);
        void Reroot(int part);                             // reverse the hinges on the path so `part` becomes the root
        int CountHinges() const;
        void RepairHinges();
        bool Working(int part) const;                      // has a hinge that actually turns
        int HingedAncestor(int part) const;                // nearest ancestor part with a working hinge, -1 if none
        int Master() const;                                // the hinge the latch holds
        std::vector<int> ReleaseCandidates(int part) const;  // hinges that may carry this hinge's catch
        bool SetTrigger(int part, int trigger);            // false if it would make a release loop
        void DefaultReleases();                            // each hinge released by the one it hangs from
        void RepairReleases();
        ReleasePlan Plan() const;
        float FoldFactor(int part, float t, const ReleasePlan& plan) const;   // share of the closed angle at fold t
        std::string Serialize() const;
        bool Deserialize(const std::string& text, std::string* err);
        bool operator==(const Design& o) const;
        bool operator!=(const Design& o) const { return !(*this == o); }
    };
}
