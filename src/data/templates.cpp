#include "design.hpp"
#include "util/utils.hpp"
#include <raymath.h>
#include <cmath>

namespace Bakugan {
    namespace {
        SuperFrustum SF(Vector3 pos, Vector3 rotDeg, Vector3 size, float rounding = 0.0f, float taper = 1.0f, float dilation = 0.0f) {
            SuperFrustum f;
            f.position = pos;
            f.rotation = Utils::EulerDegreesToQuaternion(rotDeg);
            f.size = size;
            f.rounding = rounding;
            f.taper = taper;
            f.dilation = dilation;
            return f;
        }

        // Sweep axis pointing up (+Y): size is width (x), depth (z), height (y).
        SuperFrustum Up(Vector3 pos, float w, float d, float h, float rounding = 0.0f, float taper = 1.0f) {
            return SF(pos, { -90, 0, 0 }, { w, d, h }, rounding, taper);
        }

        int Body(Design& d, const char* name) {
            return d.groups[d.AddGroup(name, -1)].id;
        }

        int Feature(Design& d, const char* name, int parent, Op op = Op::Union, float blend = 0.0f) {
            int g = d.AddGroup(name, parent);
            d.groups[g].op = op;
            d.groups[g].blend = blend;
            return d.groups[g].id;
        }

        Hinge& AddHinge(Design& d, int parent, int child, Vector3 pivot, Vector3 axis, float startDeg, float endDeg) {
            Hinge& h = d.hinges[d.AddHinge(parent, child)];
            h.pivot = pivot;
            h.axis = axis;
            h.start = startDeg * DEG2RAD;
            h.end = endDeg * DEG2RAD;
            return h;
        }

        // A box with a lid on a door hinge along the back edge, modelled closed. withLip adds a lip under the lid that is
        // a little too deep, so it digs into the back wall while the lid closes.
        Design HingedBox(bool withLip) {
            Design d;
            int base = Body(d, "Base");
            d.AddShape(Up({ 0, 15, 0 }, 60, 40, 30, 0.12f), base, "Base shell");
            d.AddShape(Up({ 0, 17.5f, 0 }, 56, 36, 30, 0.1f), base, "Cavity", Op::Cut);
            int lid = Body(d, "Lid");
            d.AddShape(Up({ 0, 31.5f, 0 }, 60, 40, 3, 0.12f), lid, "Lid plate");
            int knob = Feature(d, "Knob", lid, Op::Union, 1.5f);
            d.AddShape(Up({ 0, 36, 13 }, 6, 6, 6, 1.0f), knob, "Knob stem");
            d.AddShape(SF({ 0, 40, 13 }, { 0, 0, 0 }, { 0.2f, 0.2f, 0.2f }, 1.0f, 1.0f, 4.5f), knob, "Knob ball");
            if (withLip) {
                int lip = Feature(d, "Lip", lid);
                d.AddShape(Up({ 0, 28.5f, -0.45f }, 54, 35.9f, 3, 0.1f), lip, "Lip ring");
                d.AddShape(Up({ 0, 28.5f, -0.45f }, 50, 31.9f, 3.2f, 0.1f), lip, "Lip hollow", Op::Cut);
            }
            Hinge& h = AddHinge(d, base, lid, { 0, 31, -23 }, { 1, 0, 0 }, -105, 0);
            h.length = 40;
            h.knuckles = 5;
            h.leafLength = 5;
            return d;
        }

        // Two rounded half shells on a hinge along the back face, like a compact case. The leaves lie flat on the back
        // face (manual leaf directions: down into the bottom shell, up into the top one).
        Design Clamshell() {
            Design d;
            int bottom = Body(d, "Bottom shell");
            d.AddShape(Up({ 0, 8, 0 }, 50, 50, 16, 0.5f), bottom, "Bottom outer");
            d.AddShape(Up({ 0, 10, 0 }, 46, 46, 16, 0.5f), bottom, "Bottom hollow", Op::Cut);
            int clip = Feature(d, "Catch", bottom, Op::Union, 0.8f);
            d.AddShape(SF({ 0, 12.25f, 25.6f }, { 0, 0, 0 }, { 10, 6.5f, 1.6f }, 0.3f), clip, "Catch tab");
            d.AddShape(SF({ 0, 13, 26.6f }, { 0, 0, 0 }, { 10, 1.4f, 1.4f }, 1.0f), clip, "Catch bead");
            int top = Body(d, "Top shell");
            d.AddShape(Up({ 0, 23, 0 }, 50, 50, 14, 0.5f), top, "Top outer");
            d.AddShape(Up({ 0, 21, 0 }, 46, 46, 14, 0.5f), top, "Top hollow", Op::Cut);
            Hinge& h = AddHinge(d, bottom, top, { 0, 16, -26 }, { 1, 0, 0 }, -125, 0);
            h.length = 30;
            h.knuckles = 3;
            h.leafLength = 8;
            h.leafThickness = 2.4f;
            h.autoLeaves = false;
            h.leafAngle[0] = 90 * DEG2RAD;
            h.leafAngle[1] = -90 * DEG2RAD;
            return d;
        }

        // Four slats in a row, folding into a zig-zag stack one hinge after the other (180 degrees each).
        Design FoldingRuler() {
            Design d;
            const float len = 40.0f, thick = 4.0f, width = 14.0f, gap = 0.5f;
            int prev = -1;
            for (int i = 0; i < 4; i++) {
                std::string name = "Slat " + std::to_string(i + 1);
                int b = Body(d, name.c_str());
                d.AddShape(SF({ i * len, thick * 0.5f, 0 }, { 0, 0, 0 }, { len, thick, width }, 0.0f), b, name + " bar");
                if (prev >= 0) {
                    bool top = (i % 2) == 1;
                    float x = i * len - len * 0.5f;
                    Vector3 pivot = { x, top ? thick + gap * 0.5f : -gap * 0.5f, 0 };
                    Hinge& h = AddHinge(d, prev, b, pivot, { 0, 0, 1 }, 0, top ? 180 : -180);
                    h.length = width - 2;
                    h.knuckles = 3;
                    h.radius = 2.2f;
                    h.pinRadius = 0.9f;
                    h.clearance = 0.35f;
                    h.leafLength = 0.0f;
                    h.t0 = (i - 1) / 3.0f;
                    h.t1 = i / 3.0f;
                }
                prev = b;
            }
            return d;
        }
    }

    namespace {
        // Two flaps folding over a base from opposite sides, modelled closed: the inner one closes first and the outer
        // one over it. Opening has to wait for the outer flap to clear before the inner one lifts, so the inner hinge
        // has its own opening timing with a delay.
        Design OverlappingFlaps() {
            Design d;
            const float thick = 3.0f, gap = 0.6f;
            const float innerY = 4 + gap + thick * 0.5f;              // panel centres
            const float outerY = innerY + thick + gap;
            int base = Body(d, "Base");
            d.AddShape(Up({ 0, 2, 0 }, 44, 36, 4, 0.1f), base, "Plate");
            d.AddShape(Up({ 21, (4 + gap + thick) * 0.5f, 0 }, 6, 36, 4 + gap + thick, 0.1f), base, "Riser");
            int inner = Body(d, "Inner flap");
            d.AddShape(Up({ -4, innerY, 0 }, 36, 32, thick, 0.1f), inner, "Inner panel");
            int outer = Body(d, "Outer flap");
            d.AddShape(Up({ 3, outerY, 0 }, 38, 32, thick, 0.1f), outer, "Outer panel");
            AddHinge(d, base, inner, { -22, innerY, 0 }, { 0, 0, 1 }, 150, 0).name = "Inner hinge";
            AddHinge(d, base, outer, { 22, outerY, 0 }, { 0, 0, 1 }, -150, 0).name = "Outer hinge";
            for (Hinge& h : d.hinges) {
                h.length = 30.0f;
                h.knuckles = 5;
                h.radius = 2.6f;
                h.pinRadius = 1.0f;
                h.clearance = 0.4f;
                h.leafLength = 0.0f;
            }
            Hinge& a = d.hinges[0];
            Hinge& b = d.hinges[1];
            // closing: the inner flap first, then the outer one over it
            a.t0 = 0.0f;  a.t1 = 0.45f;
            b.t0 = 0.45f; b.t1 = 1.0f;
            // opening: the outer flap lifts first; the inner one waits until it is out of the way
            b.ownOpening = true; b.o0 = 0.0f;  b.o1 = 0.6f;
            a.ownOpening = true; a.o0 = 0.35f; a.o1 = 1.0f;
            return d;
        }
    }

    std::vector<std::string> TemplateNames() {
        return { "Hinged box", "Clamshell", "Folding ruler (sequenced)", "Collision demo (lip too deep)",
                 "Overlapping flaps (delayed opening)", "Empty" };
    }

    Design MakeTemplate(int index) {
        Design d;
        switch (index) {
            case 0: d = HingedBox(false); break;
            case 1: d = Clamshell(); break;
            case 2: d = FoldingRuler(); break;
            case 3: d = HingedBox(true); break;
            case 4: d = OverlappingFlaps(); break;
            default: break;
        }
        d.Repair();
        return d;
    }
}
