#include "suite_draw.hpp"
#include "data/field.hpp"
#include <raymath.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>

namespace Bakugan::Suite {
    namespace {
#include "suitefont.inc"

        bool FontPixel(int x, int y) {
            int i = y * 128 + x;
            return (kFontBits[i / 32] >> (i % 32)) & 1u;
        }

        const FontGlyph& Glyph(char ch) {
            unsigned char c = (unsigned char)ch;
            if (c < 32 || c > 126) c = '?';
            return kFontGlyphs[c - 32];
        }

        // Surface nets over a sampled field: a vertex per cell the surface passes through (the mean of its edge
        // crossings), a quad per grid edge it crosses; normals from the field's gradient.
        void SurfaceNets(const std::function<float(Vector3)>& f, BoundingBox box, float h, BodyMesh* out) {
            int n[3];
            for (int a = 0; a < 3; a++) n[a] = std::max(2, (int)ceilf(((&box.max.x)[a] - (&box.min.x)[a]) / h) + 1);
            auto at = [&](int x, int y, int z) { return ((size_t)z * n[1] + y) * n[0] + x; };
            std::vector<float> v((size_t)n[0] * n[1] * n[2]);
            for (int z = 0; z < n[2]; z++)
                for (int y = 0; y < n[1]; y++)
                    for (int x = 0; x < n[0]; x++)
                        v[at(x, y, z)] = f({ box.min.x + x * h, box.min.y + y * h, box.min.z + z * h });
            std::vector<int> vert((size_t)n[0] * n[1] * n[2], -1);   // per cell (indexed by its lowest corner)
            for (int z = 0; z + 1 < n[2]; z++)
                for (int y = 0; y + 1 < n[1]; y++)
                    for (int x = 0; x + 1 < n[0]; x++) {
                        float c[8];
                        int inside = 0;
                        for (int k = 0; k < 8; k++) {
                            c[k] = v[at(x + (k & 1), y + ((k >> 1) & 1), z + ((k >> 2) & 1))];
                            inside += c[k] < 0.0f;
                        }
                        if (inside == 0 || inside == 8) continue;
                        Vector3 sum = { 0, 0, 0 };
                        int count = 0;
                        for (int a = 0; a < 3; a++)
                            for (int k = 0; k < 8; k++) {
                                if (k & (1 << a)) continue;
                                int k2 = k | (1 << a);
                                if ((c[k] < 0.0f) == (c[k2] < 0.0f)) continue;
                                float t = c[k] / (c[k] - c[k2]);
                                Vector3 p = { (float)(k & 1), (float)((k >> 1) & 1), (float)((k >> 2) & 1) };
                                (&p.x)[a] = t;
                                sum = Vector3Add(sum, p);
                                count++;
                            }
                        Vector3 local = Vector3Scale(sum, 1.0f / count);
                        Vector3 g = { (c[1] - c[0] + c[3] - c[2] + c[5] - c[4] + c[7] - c[6]),
                                      (c[2] - c[0] + c[3] - c[1] + c[6] - c[4] + c[7] - c[5]),
                                      (c[4] - c[0] + c[5] - c[1] + c[6] - c[2] + c[7] - c[3]) };
                        float gl = Vector3Length(g);
                        vert[at(x, y, z)] = (int)out->pos.size();
                        out->pos.push_back({ box.min.x + (x + local.x) * h, box.min.y + (y + local.y) * h, box.min.z + (z + local.z) * h });
                        out->nrm.push_back(gl > 0.0f ? Vector3Scale(g, 1.0f / gl) : Vector3{ 0, 1, 0 });
                    }
            for (int z = 0; z < n[2]; z++)
                for (int y = 0; y < n[1]; y++)
                    for (int x = 0; x < n[0]; x++) {
                        int p[3] = { x, y, z };
                        for (int a = 0; a < 3; a++) {
                            int u = (a + 1) % 3, w = (a + 2) % 3;
                            if (p[a] + 1 >= n[a] || p[u] < 1 || p[w] < 1 || p[u] + 1 >= n[u] || p[w] + 1 >= n[w]) continue;
                            int q[3] = { x, y, z };
                            q[a]++;
                            bool in0 = v[at(x, y, z)] < 0.0f, in1 = v[at(q[0], q[1], q[2])] < 0.0f;
                            if (in0 == in1) continue;
                            // the four cells around the edge
                            int cells[4][3];
                            for (int k = 0; k < 4; k++) {
                                cells[k][0] = x; cells[k][1] = y; cells[k][2] = z;
                            }
                            cells[0][u]--; cells[0][w]--;
                            cells[1][w]--;
                            cells[3][u]--;
                            int id[4];
                            bool ok = true;
                            for (int k = 0; k < 4 && ok; k++) {
                                id[k] = vert[at(cells[k][0], cells[k][1], cells[k][2])];
                                ok = id[k] >= 0;
                            }
                            if (!ok) continue;
                            if (in0) std::swap(id[1], id[3]);
                            out->idx.insert(out->idx.end(), { (unsigned)id[0], (unsigned)id[1], (unsigned)id[2],
                                                              (unsigned)id[0], (unsigned)id[2], (unsigned)id[3] });
                        }
                    }
        }

        Rgb Lit(Rgb base, Vector3 n, const View& v) {
            float diffuse = fmaxf(0.0f, Vector3DotProduct(n, v.light));
            float facing = fabsf(Vector3DotProduct(n, v.fwd));
            Vector3 half = Vector3Normalize(Vector3Subtract(v.light, v.fwd));
            float spec = powf(fmaxf(0.0f, Vector3DotProduct(n, half)), 24.0f) * 0.25f;
            float k = (0.42f + 0.58f * diffuse) * (0.8f + 0.2f * facing);
            auto ch = [&](unsigned char c) { return (unsigned char)std::clamp(lroundf(c * k + 255.0f * spec), 0L, 255L); };
            return { ch(base.r), ch(base.g), ch(base.b) };
        }
    }

    Rgb Mix(Rgb a, Rgb b, float t) {
        t = std::clamp(t, 0.0f, 1.0f);
        return { (unsigned char)lroundf(a.r + (b.r - a.r) * t), (unsigned char)lroundf(a.g + (b.g - a.g) * t),
                 (unsigned char)lroundf(a.b + (b.b - a.b) * t) };
    }

    // ---- canvas -------------------------------------------------------------------------------------------------------

    void Canvas::Fill(int x0, int y0, int x1, int y1, Rgb c) {
        x0 = std::max(x0, 0); y0 = std::max(y0, 0); x1 = std::min(x1, w); y1 = std::min(y1, h);
        for (int y = y0; y < y1; y++)
            for (int x = x0; x < x1; x++) px[(size_t)y * w + x] = c;
    }

    void Canvas::Frame(int x0, int y0, int x1, int y1, int t, Rgb c) {
        Fill(x0, y0, x1, y0 + t, c);
        Fill(x0, y1 - t, x1, y1, c);
        Fill(x0, y0, x0 + t, y1, c);
        Fill(x1 - t, y0, x1, y1, c);
    }

    void Canvas::Text(int x, int y, int scale, const std::string& s, Rgb c) {
        for (char ch : s) {
            const FontGlyph& g = Glyph(ch);
            for (int gy = 0; gy < 10; gy++)
                for (int gx = 0; gx < g.w; gx++)
                    if (FontPixel(g.x + gx, g.y + gy)) Fill(x + gx * scale, y + gy * scale, x + (gx + 1) * scale, y + (gy + 1) * scale, c);
            x += (g.w + 1) * scale;
        }
    }

    void Canvas::Rgb24(std::vector<unsigned char>* out) const {
        out->resize(px.size() * 3);
        for (size_t k = 0; k < px.size(); k++) {
            (*out)[k * 3] = px[k].r;
            (*out)[k * 3 + 1] = px[k].g;
            (*out)[k * 3 + 2] = px[k].b;
        }
    }

    int TextWidth(const std::string& s, int scale) {
        int w = 0;
        for (char ch : s) w += (Glyph(ch).w + 1) * scale;
        return std::max(0, w - scale);
    }

    std::string FitText(std::string s, int width, int scale) {
        if (TextWidth(s, scale) <= width) return s;
        while (!s.empty() && TextWidth(s + "...", scale) > width) s.pop_back();
        return s + "...";
    }

    // ---- designs as meshes ----------------------------------------------------------------------------------------------

    DesignMesh MeshDesign(const Design& d, float voxel) {
        DesignMesh out;
        for (int g : d.Bodies()) {
            int id = d.groups[g].id;
            BodyField f = BodyField::Body(d, id);
            if (f.Empty()) continue;
            float blend = d.groups[g].blend;
            for (int c : d.ChildGroups(id)) blend = fmaxf(blend, d.groups[c].blend);
            BoundingBox box = d.BodyBounds(id);
            box.min = Vector3SubtractValue(box.min, blend + 2.0f * voxel);
            box.max = Vector3AddValue(box.max, blend + 2.0f * voxel);
            // a huge body gets coarser cells rather than an enormous grid
            Vector3 e = Vector3Subtract(box.max, box.min);
            float h = fmaxf(voxel, fmaxf(e.x, fmaxf(e.y, e.z)) / 160.0f);
            BodyMesh m;
            m.group = g;
            Color c = d.groups[g].color;
            m.color = { c.r, c.g, c.b };
            SurfaceNets([&](Vector3 p) { return f.Distance(p); }, box, h, &m);
            if (!m.idx.empty()) out.push_back(std::move(m));
        }
        return out;
    }

    // ---- camera and rasterizer ------------------------------------------------------------------------------------------

    Vector3 View::Project(Vector3 p) const {
        Vector3 q = Vector3Subtract(p, centre);
        return { ox + Vector3DotProduct(q, right) * scale, oy - Vector3DotProduct(q, up) * scale, Vector3DotProduct(q, fwd) };
    }

    View View::Framing(const std::vector<Vector3>& points, int w, int h) {
        View v;
        Vector3 eye = Vector3Normalize({ 0.62f, 0.48f, 0.9f });
        v.fwd = Vector3Negate(eye);
        v.right = Vector3Normalize(Vector3CrossProduct(v.fwd, { 0, 1, 0 }));
        v.up = Vector3CrossProduct(v.right, v.fwd);
        v.light = Vector3Normalize(Vector3Add(Vector3Add(Vector3Scale(v.right, -0.35f), Vector3Scale(v.up, 0.8f)), Vector3Scale(v.fwd, -0.5f)));
        float x0 = FLT_MAX, x1 = -FLT_MAX, y0 = FLT_MAX, y1 = -FLT_MAX;
        for (Vector3 p : points) {
            float x = Vector3DotProduct(p, v.right), y = Vector3DotProduct(p, v.up);
            x0 = fminf(x0, x); x1 = fmaxf(x1, x); y0 = fminf(y0, y); y1 = fmaxf(y1, y);
        }
        if (points.empty()) { x0 = y0 = -1; x1 = y1 = 1; }
        float sw = fmaxf(x1 - x0, 1e-3f), sh = fmaxf(y1 - y0, 1e-3f);
        v.scale = 0.94f * fminf(w / sw, h / sh);
        v.centre = Vector3Add(Vector3Scale(v.right, (x0 + x1) * 0.5f), Vector3Scale(v.up, (y0 + y1) * 0.5f));
        v.ox = w * 0.5f;
        v.oy = h * 0.5f;
        return v;
    }

    void Raster::Clear(int width, int height, const std::vector<Rgb>& background) {
        w = width;
        h = height;
        color = background;
        depth.assign((size_t)w * h, FLT_MAX);
        id.assign((size_t)w * h, 0);
    }

    void Raster::Triangle(Vector3 a, Vector3 b, Vector3 c, Rgb ca, Rgb cb, Rgb cc, unsigned char tid) {
        float area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        if (fabsf(area) < 1e-8f) return;
        int x0 = std::max(0, (int)floorf(fminf(a.x, fminf(b.x, c.x)))), x1 = std::min(w - 1, (int)ceilf(fmaxf(a.x, fmaxf(b.x, c.x))));
        int y0 = std::max(0, (int)floorf(fminf(a.y, fminf(b.y, c.y)))), y1 = std::min(h - 1, (int)ceilf(fmaxf(a.y, fmaxf(b.y, c.y))));
        float inv = 1.0f / area;
        for (int y = y0; y <= y1; y++)
            for (int x = x0; x <= x1; x++) {
                float px = x + 0.5f, py = y + 0.5f;
                float w0 = ((c.x - b.x) * (py - b.y) - (c.y - b.y) * (px - b.x)) * inv;
                float w1 = ((a.x - c.x) * (py - c.y) - (a.y - c.y) * (px - c.x)) * inv;
                float w2 = 1.0f - w0 - w1;
                if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) continue;
                float z = w0 * a.z + w1 * b.z + w2 * c.z;
                size_t i = (size_t)y * w + x;
                if (z >= depth[i]) continue;
                depth[i] = z;
                id[i] = tid;
                color[i] = { (unsigned char)lroundf(w0 * ca.r + w1 * cb.r + w2 * cc.r), (unsigned char)lroundf(w0 * ca.g + w1 * cb.g + w2 * cc.g),
                             (unsigned char)lroundf(w0 * ca.b + w1 * cb.b + w2 * cc.b) };
            }
    }

    void DrawDesign(Raster& r, const DesignMesh& m, const std::vector<Rigid>& xf, const View& v) {
        std::vector<Vector3> sp;
        std::vector<Rgb> sc;
        for (size_t b = 0; b < m.size(); b++) {
            const BodyMesh& bm = m[b];
            const Rigid& t = xf[bm.group];
            sp.resize(bm.pos.size());
            sc.resize(bm.pos.size());
            for (size_t i = 0; i < bm.pos.size(); i++) {
                sp[i] = v.Project(t.Apply(bm.pos[i]));
                sc[i] = Lit(bm.color, Vector3Normalize(t.ApplyVector(bm.nrm[i])), v);
            }
            unsigned char id = (unsigned char)(1 + b % 250);
            for (size_t i = 0; i + 2 < bm.idx.size(); i += 3)
                r.Triangle(sp[bm.idx[i]], sp[bm.idx[i + 1]], sp[bm.idx[i + 2]], sc[bm.idx[i]], sc[bm.idx[i + 1]], sc[bm.idx[i + 2]], id);
        }
        const float jump = 6.0f;     // pixels of depth
        std::vector<Rgb> out = r.color;
        for (int y = 0; y + 1 < r.h; y++)
            for (int x = 0; x + 1 < r.w; x++) {
                size_t i = (size_t)y * r.w + x, ri = i + 1, di = i + r.w;
                bool edge = r.Edge(i);
                if (!edge && r.id[i]) edge = fabsf(r.depth[i] - r.depth[ri]) * v.scale > jump || fabsf(r.depth[i] - r.depth[di]) * v.scale > jump;
                if (edge) out[i] = Mix(r.id[i] ? r.color[i] : r.color[ri], Ink, 0.7f);
            }
        r.color.swap(out);
    }

    std::vector<Rgb> Ghost(const TriMesh& m, const View& v, int w, int h) {
        const Rgb fill = { 226, 231, 240 }, outline = { 190, 198, 212 };
        Raster r;
        r.Clear(w, h, std::vector<Rgb>((size_t)w * h, White));
        std::vector<Vector3> sp(m.positions.size());
        for (size_t k = 0; k < sp.size(); k++) sp[k] = v.Project(m.positions[k]);
        for (size_t k = 0; k + 2 < m.indices.size(); k += 3) r.Triangle(sp[m.indices[k]], sp[m.indices[k + 1]], sp[m.indices[k + 2]], fill, fill, fill, 1);
        std::vector<Rgb> out = r.color;
        for (int y = 0; y + 1 < h; y++)
            for (int x = 0; x + 1 < w; x++)
                if (r.Edge((size_t)y * w + x)) out[(size_t)y * w + x] = outline;
        return out;
    }
}
