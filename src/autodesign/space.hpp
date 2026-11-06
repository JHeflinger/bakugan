#pragma once
// The design space: the limits every design the search keeps must stay inside, and the three ways a design is brought
// into them. Shared by every design of a run, so a start design outside the defaults is never stuck.
#include "autodesign.hpp"
#include "target.hpp"
#include <unordered_map>

namespace Bakugan::Auto {
    struct DesignSpace {
        Vector3 centre = { 0, 0, 0 };    // the mesh's
        float L = 1.0f;
        float reach = 0.0f;              // how far shapes and pivots may be from the centre
        float minSize = 0.0f, maxSize = 0.0f, maxDilation = 0.0f, maxBlend = 0.0f;
        int minBodies = 1, maxBodies = 1, maxShapes = 1;
        bool cuts = true, features = true;
        bool timing = false;             // hinge timing is searched (only the motion checks can see it)
        bool hingeHardware = false;      // the result's hinges get hardware ...
        std::unordered_map<int, bool> startHardware;   // ... unless the start design had it off (hinge id)

        void Init(const Target& t, const AutoParams& p);
        // Refining a design from the editor: whatever it already does is allowed, with a little room.
        void Allow(const Design& start);

        // After a structural change: repairs it, drops empty features and bodies without material; false when what
        // is left breaks a limit (the change is then thrown away).
        bool Sanitize(Design& d) const;
        // After a descent step: values are pulled back into their ranges (the paper's projection to constraints; a
        // step should be clamped, not thrown away).
        void Project(Design& d) const;
        // The result as it is handed over: hinge hardware as asked for.
        void Finalize(Design& d) const;

        bool RoomForShape(const Design& d) const { return (int)d.shapes.size() < maxShapes; }
        bool RoomForBody(const Design& d) const { return (int)d.Bodies().size() < maxBodies; }
    };
}
