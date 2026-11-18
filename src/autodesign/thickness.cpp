// The thickness part of the loss (thinWeight, minThickness): no piece thinner than can be printed and handled.
//
// A piece's thickness is taken as the diameter of the largest ball that fits inside it: twice the depth, -D, of its
// deepest point (D its field). Found on a grid over its box, then climbed towards the deepest point by steps against
// the field's gradient. A piece thinner than minThickness costs 1 - thickness / minThickness (so a piece with no
// material left costs 1), summed over the pieces. Slivers, thin caps and flat plates all show up as a shallow deepest
// point. (A thin fin on an otherwise thick piece doesn't: the ball fits in the thick part.)
//
// Its gradient: the deepest point is a maximum, so (the envelope theorem) only the field there moves the cost:
// d cost / d D(x*) = 1 / (minThickness / 2). That depends on the piece's shape, not its pose, so it goes in as an
// Anchor (what the cutters' parameters see; the joints' don't move it).
#include "loss.hpp"
#include "data/field.hpp"
#include <raymath.h>
#include <algorithm>
#include <cfloat>
#include <cmath>

namespace Bakugan {
    float Loss::Thickness(const Bodies& b, AutoScore& s, Anchors* anchors) const {
        const float half = 0.5f * std::max(1e-3f * m_L, m_P.minThickness), w = std::max(0.0f, m_P.thinWeight);
        const float h = 1e-3f * m_L;
        float sum = 0.0f, thinnest = FLT_MAX;
        for (size_t k = 0; k < b.fields.size(); k++) {
            const BodyField& f = b.fields[k];
            const BoundingBox& box = b.box[k];
            const int G = 7;
            Vector3 size = Vector3Subtract(box.max, box.min), best = box.min;
            float depth = 0.0f;
            bool found = false;
            for (int x = 0; x < G; x++)
                for (int y = 0; y < G; y++)
                    for (int z = 0; z < G; z++) {
                        Vector3 q = { box.min.x + (x + 0.5f) / G * size.x, box.min.y + (y + 0.5f) / G * size.y,
                                      box.min.z + (z + 0.5f) / G * size.z };
                        float d = -f.Distance(q);
                        if (d > depth) { depth = d; best = q; found = true; }
                    }
            if (found) {
                // climb: deeper is against the field's gradient
                float step = 0.5f * fminf(size.x, fminf(size.y, size.z)) / G;
                for (int it = 0; it < 32 && step > h; it++) {
                    auto at = [&](Vector3 d) { return f.Distance(Vector3Add(best, d)); };
                    Vector3 g = { at({ h, 0, 0 }) - at({ -h, 0, 0 }), at({ 0, h, 0 }) - at({ 0, -h, 0 }), at({ 0, 0, h }) - at({ 0, 0, -h }) };
                    if (Vector3Length(g) < 1e-12f) break;
                    Vector3 q = Vector3Subtract(best, Vector3Scale(Vector3Normalize(g), step));
                    float d = -f.Distance(q);
                    if (d > depth) { depth = d; best = q; }
                    else step *= 0.5f;
                }
            }
            thinnest = fminf(thinnest, 2.0f * depth);
            float shortfall = fmaxf(0.0f, 1.0f - depth / half);
            sum += shortfall;
            if (anchors && found && shortfall > 0.0f && w > 0.0f) {
                // The deepest point is where the ball touches two faces (or more) at once, and the field there is only
                // the nearer one's: moving either face by e changes the depth by e/2. So the slope goes to two points,
                // half way to each face, across the piece's thin way (the direction both sides get shallower fastest),
                // half to each.
                static const Vector3 dirs[] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 1, 0 }, { 1, -1, 0 }, { 1, 0, 1 }, { 1, 0, -1 },
                                                { 0, 1, 1 }, { 0, 1, -1 }, { 1, 1, 1 }, { 1, 1, -1 }, { 1, -1, 1 }, { -1, 1, 1 } };
                const float r = 0.5f * depth;
                Vector3 across = { 1, 0, 0 };
                float shallowest = -FLT_MAX;
                for (Vector3 d : dirs) {
                    Vector3 u = Vector3Normalize(d);
                    float both = f.Distance(Vector3Add(best, Vector3Scale(u, r))) + f.Distance(Vector3Subtract(best, Vector3Scale(u, r)));
                    if (both > shallowest) { shallowest = both; across = u; }
                }
                for (float side : { 1.0f, -1.0f })
                    anchors->push_back({ (int)k, Vector3Add(best, Vector3Scale(across, side * r)), 0.5f * w / half });
            }
        }
        s.thinnest = b.fields.empty() ? 0.0f : thinnest;
        return sum;
    }
}
