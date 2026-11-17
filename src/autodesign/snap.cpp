// Keeping every hinge on its pieces (Loss::SnapJoints). The hinge term in the loss only charges a joint for being
// away from its two pieces; the silhouettes and the other terms often pull harder, and the pivot drifts off. Opened,
// such a piece swings about a point in the air and floats free of the piece it hangs from: nothing could join them.
//
// So after every Adam step and every rewrite, each joint is projected back (projected gradient descent):
//   - the pivot onto the seam: where both pieces' surfaces are, closed (they share the face the cutter left).
//     Alternating projections onto the two surfaces, each a Newton step along its field's gradient, converge there.
//   - the axis along the seam: the barrel's two ends are projected too, and the axis runs between them; where they
//     don't both land, the axis is kept in the face (its component along the face's normal taken out).
//   - with hinges built flush (buildHinges, hinges.flush), the pivot is also kept deep enough for the barrel.
// A joint whose two pieces no longer touch anywhere near (a later cut came between them) is hung from another piece
// that touches its child and doesn't hang from it, so the joints stay a tree and every piece stays joined. When no piece
// touches it at all (a hollow cut through the seam, leaving a gap), the joint is counted as floating: the optimizer
// doesn't take a step or a rewrite that leaves one (autodesign.cpp), so the pieces stay joined throughout.
#include "loss.hpp"
#include "data/field.hpp"
#include <raymath.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

namespace Bakugan {
    namespace {
        Vector3 FieldGradient(const BodyField& f, Vector3 x, float h) {
            auto at = [&](Vector3 d) { return f.Distance(Vector3Add(x, d)); };
            return { (at({ h, 0, 0 }) - at({ -h, 0, 0 })) / (2.0f * h), (at({ 0, h, 0 }) - at({ 0, -h, 0 })) / (2.0f * h),
                     (at({ 0, 0, h }) - at({ 0, 0, -h })) / (2.0f * h) };
        }

        // One Newton step onto f's surface (no further than `most`)
        Vector3 OntoSurface(const BodyField& f, Vector3 x, float h, float most) {
            float d = f.Distance(x);
            Vector3 g = FieldGradient(f, x, h);
            float gg = Vector3DotProduct(g, g);
            if (gg < 1e-12f) return x;
            Vector3 step = Vector3Scale(g, -d / gg);
            float len = Vector3Length(step);
            if (len > most) step = Vector3Scale(step, most / len);
            return Vector3Add(x, step);
        }

        struct Seam {
            const BodyField *a, *b;
            float h, most, tol;
            Vector3 centre;
            float maxRadius;             // 0: no limit
            float Residual(Vector3 x) const { return fmaxf(fabsf(a->Distance(x)), fabsf(b->Distance(x))); }
            // the nearest point (about) where both surfaces are; false when the projections don't settle there
            bool Onto(Vector3& x) const {
                for (int k = 0; k < 32; k++) {
                    if (Residual(x) < tol) return true;
                    x = OntoSurface(*a, x, h, most);
                    x = OntoSurface(*b, x, h, most);
                    if (maxRadius > 0.0f) {
                        float r = Vector3Distance(x, centre);
                        if (r > maxRadius) x = Vector3Add(centre, Vector3Scale(Vector3Subtract(x, centre), maxRadius / r));
                    }
                }
                return Residual(x) < tol;
            }
        };
    }

    int Loss::SnapJoints(Carving& c) const {
        if (c.joints.empty()) return 0;
        std::vector<int> bodyOf;
        Design d = c.Build(&bodyOf);
        std::vector<BodyField> fields;
        for (size_t k = 0; k < c.pieces.size(); k++) fields.push_back(BodyField::Body(d, bodyOf[k]));
        const float R = c.radius, half = 0.5f * m_Hinge.length;
        const bool flush = m_P.buildHinges && m_P.hinges.flush;
        Seam base{ nullptr, nullptr, 1e-3f * m_L, 0.25f * R, fmaxf(0.5f * HingeSlack(), 1e-3f * m_L), c.centre,
                   flush ? m_Hinge.pivotRadius : 0.0f };

        // pieces hanging from a piece, directly or not (its own included)
        auto subtree = [&](int pieceId) {
            std::vector<char> in(c.pieces.size(), 0);
            in[c.PieceIndex(pieceId)] = 1;
            for (bool grew = true; grew;) {
                grew = false;
                for (const Joint& j : c.joints) {
                    int p = c.PieceIndex(j.parent), ch = c.PieceIndex(j.child);
                    if (p >= 0 && ch >= 0 && in[p] && !in[ch]) in[ch] = grew = true;
                }
            }
            return in;
        };
        // a few points inside a piece, spread over its box: other places to start looking for a seam from
        auto inside = [&](int pi) {
            std::vector<Vector3> out;
            BoundingBox box = c.PieceBounds(c.pieces[pi]);
            for (int x = 0; x < 4; x++)
                for (int y = 0; y < 4; y++)
                    for (int z = 0; z < 4; z++) {
                        Vector3 q = { box.min.x + (x + 0.5f) / 4.0f * (box.max.x - box.min.x), box.min.y + (y + 0.5f) / 4.0f * (box.max.y - box.min.y),
                                      box.min.z + (z + 0.5f) / 4.0f * (box.max.z - box.min.z) };
                        if (fields[pi].Distance(q) < 0.0f) out.push_back(q);
                    }
            return out;
        };

        int floating = 0;
        for (Joint& j : c.joints) {
            const int ci = c.PieceIndex(j.child), pi = c.PieceIndex(j.parent);
            if (ci < 0 || pi < 0) continue;
            // on the seam with piece `other`, nearest the pivot (from it, or else from points through the child)
            auto attach = [&](int other, Vector3* pivot) {
                Seam s = base;
                s.a = &fields[other];
                s.b = &fields[ci];
                Vector3 x = j.pivot;
                if (s.Onto(x)) { *pivot = x; return true; }
                bool found = false;
                float best = FLT_MAX;
                for (Vector3 q : inside(ci)) {
                    if (!s.Onto(q)) continue;
                    float dist = Vector3Distance(q, j.pivot);
                    if (dist < best) { best = dist; *pivot = q; found = true; }
                }
                return found;
            };
            Vector3 pivot;
            int parent = pi;
            if (!attach(pi, &pivot)) {
                // its pieces don't meet: hang it from the nearest piece that meets the child and doesn't hang from it
                std::vector<char> below = subtree(j.child);
                float best = FLT_MAX;
                parent = -1;
                for (int k = 0; k < (int)c.pieces.size(); k++) {
                    Vector3 q;
                    if (below[k] || !attach(k, &q)) continue;
                    float dist = Vector3Distance(q, j.pivot);
                    if (dist < best) { best = dist; pivot = q; parent = k; }
                }
                if (parent < 0) { floating++; continue; }
                j.parent = c.pieces[parent].id;
            }
            // the axis along the seam: the barrel's ends onto it, and the pivot between them
            Seam s = base;
            s.a = &fields[parent];
            s.b = &fields[ci];
            Vector3 a = Vector3Normalize(j.axis);
            Vector3 e0 = Vector3Subtract(pivot, Vector3Scale(a, half)), e1 = Vector3Add(pivot, Vector3Scale(a, half));
            Vector3 span = Vector3Subtract(e1, e0);
            if (s.Onto(e0) && s.Onto(e1) && Vector3Length(Vector3Subtract(e1, e0)) > 0.5f * Vector3Length(span)) {
                Vector3 along = Vector3Normalize(Vector3Subtract(e1, e0));
                if (Vector3DotProduct(along, a) < 0.0f) along = Vector3Negate(along);
                Vector3 mid = Vector3Scale(Vector3Add(e0, e1), 0.5f);
                if (s.Onto(mid)) pivot = mid;
                a = along;
            } else {
                Vector3 n = FieldGradient(fields[ci], pivot, base.h);   // the face's normal: keep the axis in the face
                if (Vector3Length(n) > 1e-6f) {
                    n = Vector3Normalize(n);
                    Vector3 t = Vector3Subtract(a, Vector3Scale(n, Vector3DotProduct(a, n)));
                    if (Vector3Length(t) > 1e-3f) a = Vector3Normalize(t);
                }
            }
            j.pivot = pivot;
            j.axis = a;
        }
        return floating;
    }
}
