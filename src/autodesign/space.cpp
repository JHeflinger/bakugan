#include "space.hpp"
#include "edits.hpp"
#include <algorithm>

namespace Bakugan::Auto {
    void DesignSpace::Init(const Target& t, const AutoParams& p) {
        centre = t.centre;
        L = t.L;
        reach = 1.2f * L;
        minSize = 0.008f * L;
        maxSize = 1.2f * L;
        maxDilation = 0.2f * L;
        maxBlend = 0.04f * L;           // bigger blends melt everything into one blob
        maxBodies = std::max(1, p.maxBodies);
        minBodies = std::clamp(p.minBodies, 1, maxBodies);
        maxShapes = p.maxShapes;
        cuts = p.allowCuts;
        features = p.allowFeatures;
        timing = p.allowTiming && p.motionChecks > 0 && p.collisionWeight > 0.0f;
        hingeHardware = p.hingeHardware;
    }

    void DesignSpace::Allow(const Design& d) {
        for (const Hinge& h : d.hinges) startHardware[h.id] = h.hardware;
        maxShapes = std::max(maxShapes, (int)d.shapes.size());
        maxBodies = std::max(maxBodies, (int)d.Bodies().size());
        minBodies = std::min(minBodies, (int)d.Bodies().size());
        for (const Shape& s : d.shapes) {
            const SuperFrustum& f = s.sf;
            reach = fmaxf(reach, Vector3Distance(f.position, centre) + 0.1f * L);
            minSize = fminf(minSize, 0.9f * MinOf(f.size));
            maxSize = fmaxf(maxSize, 1.1f * MaxOf(f.size));
            maxDilation = fmaxf(maxDilation, 1.1f * f.dilation);
        }
        for (const Group& g : d.groups) maxBlend = fmaxf(maxBlend, 1.1f * g.blend);
        for (const Hinge& h : d.hinges) reach = fmaxf(reach, Vector3Distance(h.pivot, centre) + 0.1f * L);
    }

    bool DesignSpace::Sanitize(Design& d) const {
        d.Repair();
        // empty features go, and bodies that add no material (keeping the hinge tree whole)
        for (bool again = true; again;) {
            again = false;
            for (int g = 0; g < (int)d.groups.size(); g++) {
                const Group& gr = d.groups[g];
                if (gr.parent >= 0 && d.DirectShapes(gr.id).empty() && d.ChildGroups(gr.id).empty()) {
                    d.RemoveGroup(g);
                    again = true;
                    break;
                }
            }
            if (again) continue;
            for (int g : d.Bodies()) {
                if (MaterialShapes(d, d.groups[g].id) > 0) continue;
                RemoveBody(d, d.groups[g].id);
                again = true;
                break;
            }
        }
        if (d.Bodies().empty() || d.shapes.empty()) return false;
        int bodies = (int)d.Bodies().size();
        if ((int)d.shapes.size() > maxShapes || bodies > maxBodies || bodies < minBodies) return false;
        for (const Group& g : d.groups)
            if (g.blend > maxBlend) return false;
        for (const Shape& s : d.shapes) {
            const SuperFrustum& f = s.sf;
            if (MinOf(f.size) < minSize || MaxOf(f.size) > maxSize || f.dilation > maxDilation) return false;
            if (Vector3Distance(f.position, centre) > reach) return false;
        }
        for (Hinge& h : d.hinges) {
            h.hardware = false;              // never part of the score
            if (Vector3Distance(h.pivot, centre) > reach) return false;
        }
        return true;
    }

    void DesignSpace::Project(Design& d) const {
        auto pull = [&](Vector3& p) {
            Vector3 off = Vector3Subtract(p, centre);
            float l = Vector3Length(off);
            if (l > reach) p = Vector3Add(centre, Vector3Scale(off, 0.999f * reach / l));
        };
        for (Shape& sh : d.shapes) {
            SuperFrustum& f = sh.sf;
            for (int a = 0; a < 3; a++) Coord(f.size, a) = std::clamp(Coord(f.size, a), minSize, maxSize);
            f.rounding = std::clamp(f.rounding, 0.0f, 1.0f);
            f.taper = std::clamp(f.taper, 0.0f, 1.0f);
            f.bulge = std::clamp(f.bulge, -1.0f, 1.0f);
            f.dilation = std::clamp(f.dilation, 0.0f, maxDilation);
            pull(f.position);
        }
        for (Group& g : d.groups) g.blend = std::clamp(g.blend, 0.0f, maxBlend);
        for (Hinge& h : d.hinges) {
            pull(h.pivot);
            h.axis = Vector3Length(h.axis) > 1e-6f ? Vector3Normalize(h.axis) : Vector3{ 0, 1, 0 };
            if (timing) ClampTiming(h);
        }
    }

    void DesignSpace::Finalize(Design& d) const {
        for (Hinge& h : d.hinges) {
            auto it = startHardware.find(h.id);
            h.hardware = hingeHardware && (it == startHardware.end() || it->second);
        }
        d.Repair();
    }
}
