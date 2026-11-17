#include "clash.hpp"
#include "util/box.hpp"
#include <raymath.h>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace Bakugan {
    // ---- where to look ---------------------------------------------------------------------------------------------------

    namespace {
        bool Moves(const Hinge& h) { return fabsf(h.end - h.start) > 1e-4f && h.t1 - h.t0 > 1e-5f; }

        // The share of a window's time at which the eased motion (smoothstep) has turned the share f of the angle.
        float EaseInverse(float f) { return 0.5f - sinf(asinf(1.0f - 2.0f * std::clamp(f, 0.0f, 1.0f)) / 3.0f); }
    }

    std::vector<MotionCheck> HingeChecks(const Design& d, int perHinge, float phase, bool ease) {
        std::vector<MotionCheck> out;
        if (perHinge <= 0) return out;
        phase -= floorf(phase);
        for (int h = 0; h < (int)d.hinges.size(); h++) {
            const Hinge& hg = d.hinges[h];
            if (!Moves(hg)) continue;
            float a = std::clamp(hg.t0, 0.0f, 1.0f), b = std::clamp(hg.t1, 0.0f, 1.0f);
            for (int k = 0; k < perHinge; k++) {
                float f = (k + phase) / perHinge;
                out.push_back({ a + (b - a) * (ease ? EaseInverse(f) : f), h, f });
            }
        }
        return out;
    }

    std::vector<std::vector<char>> HingeSubtrees(const Design& d) {
        const int n = (int)d.groups.size();
        // each body's parent hinge, then walk up from every body
        std::vector<int> parentHinge(n, -1);
        for (int g = 0; g < n; g++)
            if (d.groups[g].parent < 0) parentHinge[g] = d.ParentHinge(d.groups[g].id);
        std::vector<std::vector<char>> under(d.hinges.size(), std::vector<char>(n, 0));
        for (int g = 0; g < n; g++) {
            if (d.groups[g].parent >= 0) continue;
            int at = g;
            for (int guard = 0; guard < n && at >= 0 && parentHinge[at] >= 0; guard++) {
                int h = parentHinge[at];
                under[h][g] = 1;
                at = d.GroupIndex(d.hinges[h].parent);
            }
        }
        return under;
    }

    bool MovingApart(const Design& d, const std::vector<std::vector<char>>& under, int ga, int gb, float t) {
        for (size_t h = 0; h < d.hinges.size(); h++) {
            const Hinge& hg = d.hinges[h];
            if (!Moves(hg) || t < hg.t0 - 1e-6f || t > hg.t1 + 1e-6f) continue;
            if (under[h][ga] != under[h][gb]) return true;
        }
        return false;
    }

    std::vector<int> OpeningOrder(const Design& d) {
        std::vector<int> order(d.hinges.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return d.hinges[a].t1 > d.hinges[b].t1; });
        return order;
    }

    // ---- how much ----------------------------------------------------------------------------------------------------------

    void SampleOverlap(const Solid& a, const Solid& b, float cell, Vector3 jitter, Clash* out, bool keep) {
        BoundingBox wa = TransformBox(a.box, a.xf), wb = TransformBox(b.box, b.xf);
        if (!Overlap(wa, wb) || cell <= 0.0f) return;
        Vector3 lo = Vector3Max(wa.min, wb.min), hi = Vector3Min(wa.max, wb.max);
        Vector3 ext = Vector3Subtract(hi, lo);
        // the grid: cells of about `cell` (no more than 32 a side), shifted by the jitter, one more than fills the box
        int n[3];
        Vector3 h;
        for (int k = 0; k < 3; k++) {
            n[k] = std::clamp((int)ceilf((&ext.x)[k] / cell), 1, 32);
            (&h.x)[k] = fmaxf((&ext.x)[k] / n[k], 1e-6f);
        }
        const float reach = 0.5f * Vector3Length(h) * 1.25f;    // (the fields are not exact everywhere: some slack)
        const float volume = h.x * h.y * h.z;
        const BoundingBox boxA = Grow(a.box, reach), boxB = Grow(b.box, reach);
        auto add = [&](Vector3 p, float da, float db, float v) {
            float dd = fmaxf(da, db);
            out->volume += v;
            out->depth += (double)-dd * v;
            out->deepest = fmaxf(out->deepest, -dd);
            if (keep) out->samples.push_back({ p, -dd, v, da > db });
        };
        for (int k = 0; k <= n[2]; k++)
            for (int j = 0; j <= n[1]; j++)
                for (int i = 0; i <= n[0]; i++) {
                    Vector3 p = { lo.x + (i + jitter.x - 0.5f) * h.x, lo.y + (j + jitter.y - 0.5f) * h.y, lo.z + (k + jitter.z - 0.5f) * h.z };
                    Vector3 qa = a.xf.ApplyInverse(p);
                    if (!InBox(qa, boxA)) continue;
                    Vector3 qb = b.xf.ApplyInverse(p);
                    if (!InBox(qb, boxB)) continue;
                    float da = a.field->Distance(qa);
                    if (da > reach) continue;                // the cell is clear of a
                    float db = b.field->Distance(qb);
                    if (db > reach) continue;
                    if (da < -reach && db < -reach) {        // all of it is inside both
                        add(p, da, db, volume);
                        continue;
                    }
                    for (int c = 0; c < 8; c++) {            // a surface passes through it: its eight parts
                        Vector3 s = { p.x + ((c & 1) ? 0.25f : -0.25f) * h.x, p.y + ((c & 2) ? 0.25f : -0.25f) * h.y,
                                      p.z + ((c & 4) ? 0.25f : -0.25f) * h.z };
                        Vector3 sa = a.xf.ApplyInverse(s);
                        if (!InBox(sa, a.box)) continue;
                        Vector3 sb = b.xf.ApplyInverse(s);
                        if (!InBox(sb, b.box)) continue;
                        float ea = a.field->Distance(sa);
                        if (ea >= 0.0f) continue;
                        float eb = b.field->Distance(sb);
                        if (eb >= 0.0f) continue;
                        add(s, ea, eb, 0.125f * volume);
                    }
                }
    }

    BodySet MakeBodySet(const Design& d, float grow) {
        BodySet s;
        s.fieldOf.assign(d.groups.size(), -1);
        for (int g : d.Bodies()) {
            BodyField f = BodyField::Body(d, d.groups[g].id);
            if (f.Empty()) continue;
            s.fieldOf[g] = (int)s.fields.size();
            s.boxes.push_back(Grow(d.BodyBounds(d.groups[g].id), grow));
            s.fields.push_back(std::move(f));
            s.group.push_back(g);
        }
        return s;
    }

    MotionOverlap MeasureMotion(const Design& d, int perHinge, float cell, float phase) {
        MotionOverlap m;
        m.perHinge.assign(d.hinges.size(), 0.0);
        BodySet set = MakeBodySet(d, 0.5f * cell);
        const int n = (int)set.fields.size();
        if (n < 2) return m;
        std::vector<std::vector<char>> under = HingeSubtrees(d);
        std::vector<MotionCheck> checks = HingeChecks(d, perHinge, phase);
        checks.insert(checks.begin(), MotionCheck{ 0.0f, -1, 0.0f });
        for (const MotionCheck& c : checks) {
            std::vector<Rigid> xf = d.Transforms(c.t, true);
            Clash o;
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++) {
                    if (c.hinge >= 0 && !MovingApart(d, under, set.group[i], set.group[j], c.t)) continue;
                    SampleOverlap({ &set.fields[i], set.boxes[i], xf[set.group[i]] }, { &set.fields[j], set.boxes[j], xf[set.group[j]] }, cell,
                                  { 0.5f, 0.5f, 0.5f }, &o, false);
                }
            m.poses++;
            m.deepest = fmaxf(m.deepest, o.deepest);
            if (c.hinge < 0) { m.openVolume = o.volume; continue; }
            m.perHinge[c.hinge] = std::max(m.perHinge[c.hinge], o.volume);
            if (o.volume > m.worstVolume) {
                m.worstVolume = o.volume;
                m.worstT = c.t;
                m.worstHinge = c.hinge;
            }
        }
        return m;
    }
}
