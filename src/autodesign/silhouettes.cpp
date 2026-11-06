#include "silhouettes.hpp"
#include <algorithm>
#include <cfloat>
#include <cmath>

namespace Bakugan::Auto {
    // `n` directions spread evenly over the upper half of a sphere (a Fibonacci spiral): from steep top views to
    // nearly level side views. Silhouettes from opposite directions are mirror images, so a half covers them all.
    static std::vector<SilView> MakeViews(int n) {
        std::vector<SilView> out;
        for (int i = 0; i < n; i++) {
            float y = 1.0f - (i + 0.5f) / n;             // height of the eye above the centre, 1 (top) .. 0 (level)
            float r = sqrtf(fmaxf(0.0f, 1.0f - y * y)), phi = i * 2.39996323f + 0.6f;
            SilView w;
            Vector3 eye = { r * cosf(phi), y, r * sinf(phi) };
            w.dir = Vector3Negate(eye);
            Vector3 up = fabsf(w.dir.y) < 0.99f ? Vector3{ 0, 1, 0 } : Vector3{ 0, 0, -1 };
            w.u = Vector3Normalize(Vector3CrossProduct(w.dir, up));
            w.v = Vector3CrossProduct(w.u, w.dir);
            out.push_back(w);
        }
        return out;
    }

    // The mesh's outline in one view: triangles filled at 4x4 samples per pixel, then each pixel's covered share.
    static std::vector<float> RasterMesh(const TriMesh& m, const SilFrame& fr, const SilView& w) {
        const int ss = 4, n = fr.res * ss;
        std::vector<unsigned char> hit((size_t)n * n, 0);
        std::vector<Vector2> pts(m.positions.size());
        for (size_t i = 0; i < pts.size(); i++) pts[i] = Vector2Scale(fr.ToPixel(w, m.positions[i]), (float)ss);
        for (size_t t = 0; t + 2 < m.indices.size(); t += 3) {
            Vector2 a = pts[m.indices[t]], b = pts[m.indices[t + 1]], c = pts[m.indices[t + 2]];
            float area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
            if (fabsf(area) < 1e-12f) continue;
            int x0 = std::max(0, (int)floorf(fminf(a.x, fminf(b.x, c.x)))), x1 = std::min(n - 1, (int)ceilf(fmaxf(a.x, fmaxf(b.x, c.x))));
            int y0 = std::max(0, (int)floorf(fminf(a.y, fminf(b.y, c.y)))), y1 = std::min(n - 1, (int)ceilf(fmaxf(a.y, fmaxf(b.y, c.y))));
            for (int y = y0; y <= y1; y++)
                for (int x = x0; x <= x1; x++) {
                    float px = x + 0.5f, py = y + 0.5f;
                    float e0 = (b.x - a.x) * (py - a.y) - (b.y - a.y) * (px - a.x);
                    float e1 = (c.x - b.x) * (py - b.y) - (c.y - b.y) * (px - b.x);
                    float e2 = (a.x - c.x) * (py - c.y) - (a.y - c.y) * (px - c.x);
                    if ((e0 >= 0 && e1 >= 0 && e2 >= 0) || (e0 <= 0 && e1 <= 0 && e2 <= 0)) hit[(size_t)y * n + x] = 1;
                }
        }
        std::vector<float> out((size_t)fr.res * fr.res, 0.0f);
        for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++)
                if (hit[(size_t)y * n + x]) out[(size_t)(y / ss) * fr.res + x / ss] += 1.0f / (ss * ss);
        return out;
    }

    // The sphere's outline: a disc of rpx pixels about c0, anti-aliased over a pixel.
    static float Disc(float rpx, Vector2 c0, int x, int y) {
        float dx = x + 0.5f - c0.x, dy = y + 0.5f - c0.y;
        return std::clamp(rpx - sqrtf(dx * dx + dy * dy) + 0.5f, 0.0f, 1.0f);
    }

    // One body's outline in one view, rows [y0, y1): a ray per pixel along the view direction, sphere traced through
    // the body's field (in the body's own frame) inside its box. A pixel's coverage comes from the nearest the ray gets
    // to the body, or the deepest into it, over a pixel's width: 0 when it passes half a pixel outside, 1 when it goes
    // half a pixel inside, linear between. That is roughly the share of the pixel the body covers, and it changes
    // smoothly as the body moves, which the descent needs (hard 0/1 pixels would not change under a small step).
    static void RenderBody(const BodyField& f, const BoundingBox& box, const Rigid& xf, const SilFrame& fr, const SilView& w,
                           int y0, int y1, unsigned char* out) {
        const float tau = 0.5f * fr.Pixel();
        const float far = 4.0f * fr.half + Vector3Length(Vector3Subtract(box.max, box.min));
        // the view in the body's frame
        Quaternion inv = { -xf.rotation.x, -xf.rotation.y, -xf.rotation.z, xf.rotation.w };
        Vector3 dir = Vector3RotateByQuaternion(w.dir, inv), u = Vector3RotateByQuaternion(w.u, inv), v = Vector3RotateByQuaternion(w.v, inv);
        Vector3 base = xf.ApplyInverse(Vector3Subtract(fr.centre, Vector3Scale(w.dir, far)));
        for (int y = y0; y < y1; y++)
            for (int x = 0; x < fr.res; x++) {
                float sx = ((x + 0.5f) / fr.res * 2.0f - 1.0f) * fr.half, sy = (1.0f - (y + 0.5f) / fr.res * 2.0f) * fr.half;
                Vector3 o = Vector3Add(base, Vector3Add(Vector3Scale(u, sx), Vector3Scale(v, sy)));
                // the ray's stretch inside the box (slabs)
                float tn = 0.0f, tf = 2.0f * far;
                bool miss = false;
                for (int a = 0; a < 3 && !miss; a++) {
                    float oa = Coord(o, a), da = Coord(dir, a), lo = Coord(box.min, a) - tau, hi = Coord(box.max, a) + tau;
                    if (fabsf(da) < 1e-9f) { miss = oa < lo || oa > hi; continue; }
                    float t0 = (lo - oa) / da, t1 = (hi - oa) / da;
                    if (t0 > t1) std::swap(t0, t1);
                    tn = fmaxf(tn, t0);
                    tf = fminf(tf, t1);
                    miss = tn > tf;
                }
                unsigned char cover = 0;
                if (!miss) {
                    float t = tn, dmin = FLT_MAX;
                    for (int k = 0; k < 64 && t <= tf; k++) {
                        float d = f.Distance(Vector3Add(o, Vector3Scale(dir, t)));
                        dmin = fminf(dmin, d);
                        if (d < -tau) break;                       // well inside: fully covered
                        t += d > tau ? 0.9f * d : 0.5f * tau;     // sphere tracing, small steps near the surface
                    }
                    float c = std::clamp(0.5f - dmin / (2.0f * tau), 0.0f, 1.0f);
                    cover = (unsigned char)lroundf(c * 255.0f);
                }
                out[(size_t)y * fr.res + x] = cover;
            }
    }

    // FNV-1a, for the cache's keys.
    struct Hasher {
        unsigned long long h = 1469598103934665603ull;
        void Bytes(const void* p, size_t n) {
            const unsigned char* b = (const unsigned char*)p;
            for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
        }
        template <class T> void Add(const T& v) { Bytes(&v, sizeof(T)); }
    };

    // Everything a body's field depends on: its groups' operations and blends, nested as they are, and its shapes'
    // operations and values (hinge hardware is off while searching, so it never adds to a body).
    static void HashGroup(const Design& d, int groupIndex, Hasher& hs) {
        const Group& g = d.groups[groupIndex];
        hs.Add((int)g.op);
        hs.Add(g.blend);
        for (int s : d.DirectShapes(g.id)) {
            const Shape& sh = d.shapes[s];
            const SuperFrustum& f = sh.sf;
            hs.Add((int)sh.op);
            hs.Add(f.position); hs.Add(f.rotation); hs.Add(f.size);
            hs.Add(f.rounding); hs.Add(f.taper); hs.Add(f.bulge); hs.Add(f.dilation); hs.Add(f.onion);
        }
        for (int c : d.ChildGroups(g.id)) {
            hs.Add(-7);              // marks nesting
            HashGroup(d, c, hs);
        }
    }

    // ---- cache ----------------------------------------------------------------------------------------------------------

    void Silhouettes::Cache::Limit(size_t entryBytes) { m_Cap = std::max<size_t>(32, (size_t)(96u << 20) / std::max<size_t>(1, entryBytes)); }

    std::vector<unsigned char>* Silhouettes::Cache::Find(unsigned long long key) {
        auto it = m_Map.find(key);
        if (it == m_Map.end()) return nullptr;
        it->second.used = ++m_Tick;
        return &it->second.pixels;
    }

    std::vector<unsigned char>* Silhouettes::Cache::Add(unsigned long long key, size_t bytes) {
        if (m_Map.size() >= m_Cap) {
            // drop the least recently used quarter
            std::vector<std::pair<unsigned long long, unsigned long long>> age;
            for (const auto& [k, e] : m_Map) age.push_back({ e.used, k });
            std::sort(age.begin(), age.end());
            for (size_t i = 0; i < age.size() / 4; i++) m_Map.erase(age[i].second);
        }
        Entry& e = m_Map[key];
        e.used = ++m_Tick;
        e.pixels.assign(bytes, 0);
        return &e.pixels;
    }

    // ---- silhouettes ------------------------------------------------------------------------------------------------

    // Every view frames the same square around the mesh's box, a little wider than the mesh can ever reach (it is all
    // within half the box's diagonal of the centre); material beyond the frame is caught by the off-frame share.
    void Silhouettes::Prepare(const TriMesh& mesh, const BoundingBox& box, float L, int views, int res, Pool& pool) {
        m_Pool = &pool;
        m_Frame.centre = Vector3Scale(Vector3Add(box.min, box.max), 0.5f);
        m_Frame.half = 0.55f * L;
        m_Frame.res = std::clamp(res, 16, 256);
        m_Frame.views = MakeViews(std::clamp(views, 1, 24));
        m_Target.assign(m_Frame.views.size(), {});
        pool.For((int)m_Frame.views.size(), [&](int v) { m_Target[v] = RasterMesh(mesh, m_Frame, m_Frame.views[v]); });
        m_Cache.Limit(m_Frame.views.size() * m_Frame.Pixels());
    }

    // Each body's outlines in each pose come from the cache or are traced now (in parallel, eight rows at a time); a
    // pose's outline is the union of its bodies' (each pixel the most any body covers).
    void Silhouettes::Trace(const Design& d, const Probe& pr, const std::vector<Rigid>& open, const std::vector<Rigid>& closed) {
        const SilFrame& fr = m_Frame;
        const int nv = (int)fr.views.size(), px = fr.Pixels();
        const size_t bytes = (size_t)nv * px;
        struct Job {
            std::vector<unsigned char>* out;
            int body;
            Rigid xf;
        };
        std::vector<Job> jobs;
        std::vector<std::vector<unsigned char>*> layers[2];
        for (int pose = 0; pose < 2; pose++) {
            const std::vector<Rigid>& xf = pose == 0 ? open : closed;
            for (size_t i = 0; i < pr.fields.size(); i++) {
                const Rigid& r = xf[pr.group[i]];
                Hasher hs;
                HashGroup(d, pr.group[i], hs);
                hs.Add(r.rotation);
                hs.Add(r.position);
                std::vector<unsigned char>* img = m_Cache.Find(hs.h);
                if (!img) {
                    img = m_Cache.Add(hs.h, bytes);
                    jobs.push_back({ img, (int)i, r });
                }
                layers[pose].push_back(img);
            }
        }
        const int rows = 8, blocks = (fr.res + rows - 1) / rows;
        m_Pool->For((int)jobs.size() * nv * blocks, [&](int t) {
            const Job& j = jobs[t / (nv * blocks)];
            int v = (t / blocks) % nv, b = t % blocks;
            RenderBody(pr.fields[j.body], pr.box[j.body], j.xf, fr, fr.views[v], b * rows, std::min(fr.res, (b + 1) * rows),
                       j.out->data() + (size_t)v * px);
        });
        for (int pose = 0; pose < 2; pose++) {
            std::vector<unsigned char>& out = pose == 0 ? m_Last.open : m_Last.closed;
            out.assign(bytes, 0);
            for (const std::vector<unsigned char>* img : layers[pose])
                for (size_t k = 0; k < bytes; k++) out[k] = std::max(out[k], (*img)[k]);
        }
    }

    // Per view, against the target T (the mesh's outline for the open pose, the sphere's disc for the closed one),
    // with D the design's outline:
    //   covered = sum(min(T, D)) / sum(T)          the share of the target the design covers
    //   extra   = sum(max(0, D - T)) / sum(T)      the design's pixels where the target has none, by the same yardstick
    //   loss    = (1 - covered) + extra
    // fit and sphere are the mean losses over the views, plus twice the share of material outside the frame (pixels
    // there are never seen, so that material would otherwise be free). A view covered less than minCoverage costs 10
    // for each whole unit it falls short (a 10% shortfall costs 1, more than the rest of the score): steep enough that
    // no view can be traded away, but still a slope, so a design short of it can climb back (a flat "very bad" would
    // leave the search with no direction). The mesh's views only count while the goal leaves the mesh any weight, the
    // sphere's only while it gives the sphere any.
    void Silhouettes::Score(const Design& d, const Probe& pr, const std::vector<Rigid>& open, const std::vector<Rigid>& closed,
                            float offOpen, float offClosed, float sphereBias, float minCoverage, AutoScore* sc) {
        Trace(d, pr, open, closed);
        const SilFrame& fr = m_Frame;
        const int nv = (int)fr.views.size(), px = fr.Pixels();
        const float w = std::clamp(sphereBias, 0.0f, 1.0f), minCover = std::clamp(minCoverage, 0.0f, 1.0f);
        float worst = 1.0f, penalty = 0.0f;
        auto compare = [&](const unsigned char* design, const auto& target, bool counts, float* cover, float* extra) {
            double tsum = 0.0, hit = 0.0, spill = 0.0;
            for (int k = 0; k < px; k++) {
                float t = target(k), dd = design[k] * (1.0f / 255.0f);
                tsum += t;
                hit += fminf(t, dd);
                spill += fmaxf(0.0f, dd - t);
            }
            *cover = tsum > 0.0 ? (float)(hit / tsum) : 0.0f;
            *extra = tsum > 0.0 ? (float)(spill / tsum) : (spill > 0.0 ? 1.0f : 0.0f);
            if (counts) {
                worst = fminf(worst, *cover);
                penalty += 10.0f * fmaxf(0.0f, minCover - *cover);
            }
        };

        // the open pose against the mesh's outlines
        double fit = 0.0, cover = 0.0, extra = 0.0;
        for (int v = 0; v < nv; v++) {
            float c, e;
            const std::vector<float>& target = m_Target[v];
            compare(m_Last.open.data() + (size_t)v * px, [&](int k) { return target[k]; }, w < 1.0f, &c, &e);
            fit += (1.0f - c) + e;
            cover += c;
            extra += e;
        }
        sc->viewCover = (float)(cover / nv);
        sc->viewExtra = (float)(extra / nv);
        sc->fit = (float)(fit / nv) + 2.0f * offOpen;

        // the closed pose against the sphere: the same disc in every view, centred where the sphere's centre lands in
        // it (exact to a fraction of a pixel, so the score moves smoothly with the centre)
        if (sc->radius > 0.0f) {
            double sph = 0.0, scover = 0.0, sextra = 0.0;
            const float rpx = sc->radius / fr.Pixel();
            for (int v = 0; v < nv; v++) {
                Vector2 c0 = fr.ToPixel(fr.views[v], sc->centre);
                float c, e;
                compare(m_Last.closed.data() + (size_t)v * px, [&](int k) { return Disc(rpx, c0, k % fr.res, k / fr.res); }, w > 0.0f, &c, &e);
                sph += (1.0f - c) + e;
                scover += c;
                sextra += e;
            }
            sc->sphereViewCover = (float)(scover / nv);
            sc->sphereViewExtra = (float)(sextra / nv);
            sc->sphere = (float)(sph / nv) + 2.0f * offClosed;
        } else {
            sc->sphere = 2.0f;
            if (w > 0.0f) penalty += 10.0f * minCover * nv;
            worst = 0.0f;
        }
        sc->silhouettes = true;
        sc->worstView = worst;
        sc->coveragePenalty = penalty;
        sc->offFrame = offOpen + offClosed;
    }

    // A row of the open pose against the mesh, a row of the closed pose against the sphere. Dark where both cover a
    // pixel, red where only the target does (missing), blue where only the design does (extra), white where neither.
    void Silhouettes::Image(const Outlines& o, const AutoScore& s, std::vector<unsigned char>* rgba, int* w, int* h) const {
        const SilFrame& fr = m_Frame;
        const int nv = (int)fr.views.size(), px = fr.Pixels(), gap = 3;
        if (o.open.size() != (size_t)nv * px) { *w = *h = 0; rgba->clear(); return; }
        *w = nv * fr.res + (nv + 1) * gap;
        *h = 2 * fr.res + 3 * gap;
        rgba->assign((size_t)*w * *h * 4, 0);
        for (size_t k = 0; k < rgba->size(); k += 4) { (*rgba)[k] = 222; (*rgba)[k + 1] = 225; (*rgba)[k + 2] = 232; (*rgba)[k + 3] = 255; }
        const float both[3] = { 70, 74, 86 }, miss[3] = { 235, 110, 95 }, extra[3] = { 90, 140, 235 };
        const float rpx = s.radius / fr.Pixel();
        for (int row = 0; row < 2; row++)
            for (int v = 0; v < nv; v++) {
                Vector2 c0 = fr.ToPixel(fr.views[v], s.centre);
                const unsigned char* design = (row == 0 ? o.open : o.closed).data() + (size_t)v * px;
                int ox = gap + v * (fr.res + gap), oy = gap + row * (fr.res + gap);
                for (int k = 0; k < px; k++) {
                    int x = k % fr.res, y = k / fr.res;
                    float t = row == 0 ? m_Target[v][k] : (rpx > 0.0f ? Disc(rpx, c0, x, y) : 0.0f);
                    float dd = design[k] / 255.0f, ov = fminf(t, dd), ms = t - ov, ex = dd - ov, rest = 1.0f - fmaxf(t, dd);
                    unsigned char* out = &(*rgba)[((size_t)(oy + y) * *w + ox + x) * 4];
                    for (int c = 0; c < 3; c++)
                        out[c] = (unsigned char)std::clamp(lroundf(ov * both[c] + ms * miss[c] + ex * extra[c] + rest * 255.0f), 0L, 255L);
                }
            }
    }
}
