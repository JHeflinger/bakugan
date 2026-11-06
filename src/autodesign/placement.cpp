// Where the grammar puts new material, cuts and hinges: aimed at the spots the last score found wrong.
#include "grammar.hpp"
#include "edits.hpp"
#include "util/utils.hpp"
#include <algorithm>
#include <cfloat>

namespace Bakugan::Auto {
    // A random shape around `at`: box-ish, a few percent of the mesh's size (cuts a little smaller).
    SuperFrustum Grammar::NewShape(Vector3 at, bool cut) {
        const float L = m_T.L;
        SuperFrustum f;
        f.position = at;
        f.rotation = RandomRotation(m_Rng);
        float lo = cut ? 0.04f : 0.05f, hi = cut ? 0.14f : 0.22f;
        f.size = { L * Uniform(m_Rng, lo, hi), L * Uniform(m_Rng, lo, hi), L * Uniform(m_Rng, lo, hi) };
        f.rounding = Uniform(m_Rng);
        f.taper = Chance(m_Rng, 0.7f) ? 1.0f : Uniform(m_Rng, 0.2f, 1.0f);
        return f;
    }

    // Somewhere worth adding material (or cutting it), in the design pose, and the body (group index) it belongs to.
    // The fit and the sphere each nominate spots, as often as they matter in the score (the goal slider).
    bool Grammar::PlaceSpot(const DrawContext& c, bool forCut, Vector3* at, int* body, bool* sphere) {
        const Design& d = c.d;
        float w = std::clamp(m_P.sphereBias, 0.0f, 1.0f);
        const std::vector<Spot>& fit = forCut ? c.diag.excess : c.diag.missing;
        const std::vector<Spot>& sph = forCut ? c.diag.bumps : c.diag.holes;
        bool useSphere = !sph.empty() && (fit.empty() || Chance(m_Rng, 0.15f + 0.7f * w));
        const std::vector<Spot>& from = useSphere ? sph : fit;
        *sphere = false;
        if (from.empty()) {
            // nothing to aim at: anywhere inside the mesh
            if (forCut || m_T.inside.empty()) return false;
            *at = PickFrom(m_Rng, m_T.inside);
            *body = NearestBody(d, *at);
            return *body >= 0;
        }
        *sphere = useSphere;
        Spot s = PickFrom(m_Rng, from);
        int g = s.body;
        if (g < 0 || g >= (int)d.groups.size() || d.groups[g].parent >= 0) {
            // no body near it: the nearest body in the pose the spot is in
            if (useSphere) {
                std::vector<Rigid> closed = d.Transforms(1.0f, false);
                float best = FLT_MAX;
                for (int b : d.Bodies()) {
                    BodyField f = BodyField::Body(d, d.groups[b].id);
                    if (f.Empty()) continue;
                    float dist = f.Distance(closed[b].ApplyInverse(s.p));
                    if (dist < best) { best = dist; g = b; }
                }
            } else {
                g = NearestBody(d, s.p);
            }
        }
        if (g < 0) return false;
        std::vector<Rigid> xf = d.Transforms(useSphere ? 1.0f : 0.0f, false);
        *at = xf[g].ApplyInverse(s.p);
        *body = g;
        return true;
    }

    // A superfrustum for the region around a bad spot: points in a ball around it that are in the mesh but not the
    // design (or in the design but not the mesh, for a cut), or in the sphere but not the closed shape (or the other
    // way round), get an oriented box fitted to them, in the design frame of the body it goes into. Too few points:
    // a random shape there.
    bool Grammar::FitRegion(const DrawContext& c, bool cut, SuperFrustum* f, int* body) {
        const Design& d = c.d;
        Vector3 at;
        bool sphere = false;
        if (!PlaceSpot(c, cut, &at, body, &sphere)) return false;
        const float L = m_T.L;
        if (sphere && c.score.radius <= 0.0f) sphere = false;
        std::vector<Rigid> xf = d.Transforms(sphere ? 1.0f : 0.0f, false);
        Vector3 centre = xf[*body].Apply(at);       // the spot in its pose
        Probe pr;
        pr.Build(d, 0.002f * L);
        float rho = L * Uniform(m_Rng, 0.06f, 0.22f);
        std::vector<Vector3> keep;
        for (int i = 0; i < 400; i++) {
            Vector3 p = Vector3Add(centre, Vector3Scale(RandomInBall(m_Rng), rho));
            int who = -1;
            float dd = pr.Distance(p, &xf, 0.1f * L, &who);
            bool inTarget = sphere ? Vector3Distance(p, c.score.centre) < c.score.radius : m_T.Inside(p);
            bool want = cut ? (!inTarget && dd < 0.0f && who == *body) : (inTarget && dd > 0.0f);
            if (want) keep.push_back(xf[*body].ApplyInverse(p));
        }
        if (keep.size() < 12) {
            *f = NewShape(at, cut);
            return true;
        }
        *f = BoxAround(keep, 0.04f, 0.015f * L);
        // smaller than the region: a rewrite should change the shape only a little (the paper's jump continuity), and
        // the descent grows it from inside the gap rather than shrinking it back out of the neighbours
        f->size = Vector3Scale(f->size, Uniform(m_Rng, 0.5f, 0.8f));
        f->rounding = Chance(m_Rng, 0.5f) ? Uniform(m_Rng, 0.4f, 1.0f) : Uniform(m_Rng);
        return true;
    }

    // How a new hinge turns its child: half the time folding it towards the middle of the mesh (where the sphere wants
    // to be), about the axis that swings it there, most of the way; otherwise about a random axis across the child's
    // direction, by a random angle.
    void Grammar::ChooseHinge(Vector3 pivot, Vector3 childPoint, Vector3* axis, float* angle) {
        Vector3 out = Vector3Subtract(childPoint, pivot);
        if (Vector3Length(out) < 1e-4f) out = RandomUnit(m_Rng);
        out = Vector3Normalize(out);
        Vector3 in = Vector3Subtract(m_T.centre, pivot);
        Vector3 swing = Vector3CrossProduct(out, in);
        if (Chance(m_Rng, 0.5f) && Vector3Length(swing) > 1e-3f * Vector3Length(in)) {
            *axis = Vector3Normalize(swing);
            float cosang = Vector3DotProduct(out, Vector3Normalize(in));
            *angle = acosf(std::clamp(cosang, -1.0f, 1.0f)) * Uniform(m_Rng, 0.6f, 1.0f);
        } else {
            Vector3 r = RandomUnit(m_Rng);
            r = Vector3Subtract(r, Vector3Scale(out, Vector3DotProduct(r, out)));
            *axis = Vector3Length(r) > 1e-3f ? Vector3Normalize(r) : Utils::AnyPerpendicular(out);
            *angle = Uniform(m_Rng, -PI, PI);
        }
    }

    int Grammar::AttachHinge(Design& d, int parent, int child, Vector3 pivot, Vector3 childPoint) {
        if (!d.CanHinge(parent, child)) return -1;
        int h = d.AddHinge(parent, child);
        Vector3 axis;
        float angle;
        ChooseHinge(pivot, childPoint, &axis, &angle);
        SetupHinge(d, h, pivot, axis, angle, true, m_T.L);
        return d.hinges[h].id;
    }
}
