#include "suite.hpp"
#include "data/field.hpp"
#include "util/utils.hpp"
#include <raymath.h>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>
#ifdef _WIN32
#define SUITE_POPEN _popen
#define SUITE_PCLOSE _pclose
#define SUITE_NULL "NUL"
#else
#include <csignal>
#define SUITE_POPEN popen
#define SUITE_PCLOSE pclose
#define SUITE_NULL "/dev/null"
#endif

namespace Bakugan {
    namespace fs = std::filesystem;

    namespace {
#include "suitefont.inc"

        // ---- small helpers -------------------------------------------------------------------------------------

        // Calls f(i) for i in [0, n) on up to `threads` threads.
        template <class F>
        void ParallelFor(int n, int threads, const F& f) {
            threads = std::max(1, std::min(threads, n));
            if (threads == 1) {
                for (int i = 0; i < n; i++) f(i);
                return;
            }
            std::atomic<int> next{ 0 };
            std::vector<std::thread> pool;
            for (int t = 0; t < threads; t++)
                pool.emplace_back([&] {
                    for (int i = next++; i < n; i = next++) f(i);
                });
            for (std::thread& t : pool) t.join();
        }

        int Threads() { return std::max(1, (int)std::thread::hardware_concurrency()); }

        struct Rgb {
            unsigned char r = 255, g = 255, b = 255;
        };

        Rgb Mix(Rgb a, Rgb b, float t) {
            t = std::clamp(t, 0.0f, 1.0f);
            return { (unsigned char)lroundf(a.r + (b.r - a.r) * t), (unsigned char)lroundf(a.g + (b.g - a.g) * t),
                     (unsigned char)lroundf(a.b + (b.b - a.b) * t) };
        }

        const Rgb White = { 255, 255, 255 };
        const Rgb Ink = { 40, 42, 48 };
        const Rgb Muted = { 120, 124, 134 };
        const Rgb Border = { 208, 211, 218 };
        const Rgb Accent = { 70, 140, 230 };

        // ---- the canvas and its text -----------------------------------------------------------------------------

        struct Canvas {
            int w = 0, h = 0;
            std::vector<Rgb> px;
            void Resize(int width, int height) { w = width; h = height; px.assign((size_t)w * h, White); }
            void Fill(int x0, int y0, int x1, int y1, Rgb c) {
                x0 = std::max(x0, 0); y0 = std::max(y0, 0); x1 = std::min(x1, w); y1 = std::min(y1, h);
                for (int y = y0; y < y1; y++)
                    for (int x = x0; x < x1; x++) px[(size_t)y * w + x] = c;
            }
            void Frame(int x0, int y0, int x1, int y1, int t, Rgb c) {
                Fill(x0, y0, x1, y0 + t, c);
                Fill(x0, y1 - t, x1, y1, c);
                Fill(x0, y0, x0 + t, y1, c);
                Fill(x1 - t, y0, x1, y1, c);
            }
        };

        bool FontPixel(int x, int y) {
            int i = y * 128 + x;
            return (kFontBits[i / 32] >> (i % 32)) & 1u;
        }

        const FontGlyph& Glyph(char ch) {
            unsigned char c = (unsigned char)ch;
            if (c < 32 || c > 126) c = '?';
            return kFontGlyphs[c - 32];
        }

        int TextWidth(const std::string& s, int scale) {
            int w = 0;
            for (char ch : s) w += (Glyph(ch).w + 1) * scale;
            return std::max(0, w - scale);
        }

        void DrawText(Canvas& cv, int x, int y, int scale, const std::string& s, Rgb c) {
            for (char ch : s) {
                const FontGlyph& g = Glyph(ch);
                for (int gy = 0; gy < 10; gy++)
                    for (int gx = 0; gx < g.w; gx++)
                        if (FontPixel(g.x + gx, g.y + gy)) cv.Fill(x + gx * scale, y + gy * scale, x + (gx + 1) * scale, y + (gy + 1) * scale, c);
                x += (g.w + 1) * scale;
            }
        }

        // Cuts a string to fit, with an ellipsis.
        std::string Fit(std::string s, int width, int scale) {
            if (TextWidth(s, scale) <= width) return s;
            while (!s.empty() && TextWidth(s + "...", scale) > width) s.pop_back();
            return s + "...";
        }

        // ---- meshing and drawing designs -------------------------------------------------------------------------

        struct BodyMesh {
            int group = -1;              // group index (its transform)
            Rgb color;
            std::vector<Vector3> pos, nrm;
            std::vector<unsigned> idx;
        };
        using DesignMesh = std::vector<BodyMesh>;

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

        // A fixed orthographic camera for one box of the video, looking down at the model from the front right.
        struct View {
            Vector3 centre = { 0, 0, 0 }, right = { 1, 0, 0 }, up = { 0, 1, 0 }, fwd = { 0, 0, -1 };
            Vector3 light = { 0, 1, 0 };
            float scale = 1.0f;          // pixels per mm
            float ox = 0.0f, oy = 0.0f;  // pixel position of the centre
            Vector3 Project(Vector3 p) const {
                Vector3 q = Vector3Subtract(p, centre);
                return { ox + Vector3DotProduct(q, right) * scale, oy - Vector3DotProduct(q, up) * scale, Vector3DotProduct(q, fwd) };
            }
        };

        View MakeView(const std::vector<Vector3>& points, int w, int h) {
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

        // One box's picture at twice the output resolution, depth tested; ids mark which body covers a pixel.
        struct Raster {
            int w = 0, h = 0;
            std::vector<Rgb> color;
            std::vector<float> depth;
            std::vector<unsigned char> id;
            void Clear(const std::vector<Rgb>& background) {
                color = background;
                depth.assign((size_t)w * h, FLT_MAX);
                id.assign((size_t)w * h, 0);
            }
        };

        void DrawTriangle(Raster& r, Vector3 a, Vector3 b, Vector3 c, Rgb ca, Rgb cb, Rgb cc, unsigned char id) {
            float area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
            if (fabsf(area) < 1e-8f) return;
            int x0 = std::max(0, (int)floorf(fminf(a.x, fminf(b.x, c.x)))), x1 = std::min(r.w - 1, (int)ceilf(fmaxf(a.x, fmaxf(b.x, c.x))));
            int y0 = std::max(0, (int)floorf(fminf(a.y, fminf(b.y, c.y)))), y1 = std::min(r.h - 1, (int)ceilf(fmaxf(a.y, fmaxf(b.y, c.y))));
            float inv = 1.0f / area;
            for (int y = y0; y <= y1; y++)
                for (int x = x0; x <= x1; x++) {
                    float px = x + 0.5f, py = y + 0.5f;
                    float w0 = ((c.x - b.x) * (py - b.y) - (c.y - b.y) * (px - b.x)) * inv;
                    float w1 = ((a.x - c.x) * (py - c.y) - (a.y - c.y) * (px - c.x)) * inv;
                    float w2 = 1.0f - w0 - w1;
                    if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) continue;
                    float z = w0 * a.z + w1 * b.z + w2 * c.z;
                    size_t i = (size_t)y * r.w + x;
                    if (z >= r.depth[i]) continue;
                    r.depth[i] = z;
                    r.id[i] = id;
                    r.color[i] = { (unsigned char)lroundf(w0 * ca.r + w1 * cb.r + w2 * cc.r),
                                   (unsigned char)lroundf(w0 * ca.g + w1 * cb.g + w2 * cc.g),
                                   (unsigned char)lroundf(w0 * ca.b + w1 * cb.b + w2 * cc.b) };
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
                    DrawTriangle(r, sp[bm.idx[i]], sp[bm.idx[i + 1]], sp[bm.idx[i + 2]], sc[bm.idx[i]], sc[bm.idx[i + 1]],
                                 sc[bm.idx[i + 2]], id);
            }
            // outlines where the body under a pixel changes, or the depth jumps
            float jump = 6.0f;
            std::vector<Rgb> out = r.color;
            for (int y = 0; y + 1 < r.h; y++)
                for (int x = 0; x + 1 < r.w; x++) {
                    size_t i = (size_t)y * r.w + x, ri = i + 1, di = i + r.w;
                    bool edge = r.id[i] != r.id[ri] || r.id[i] != r.id[di];
                    if (!edge && r.id[i]) edge = fabsf(r.depth[i] - r.depth[ri]) * v.scale > jump || fabsf(r.depth[i] - r.depth[di]) * v.scale > jump;
                    if (edge) out[i] = Mix(r.id[i] ? r.color[i] : r.color[ri], Ink, 0.7f);
                }
            r.color.swap(out);
        }

        // ---- one mesh's run --------------------------------------------------------------------------------------

        struct Entry {
            std::string name;
            TriMesh mesh;
            std::vector<AutoFrame> frames;
            Design result;
            AutoScore score;
            int iterations = 0;
            std::string error;
            // rendering
            View view;
            std::vector<Rgb> ghost;      // the input mesh, faint, on white (raster resolution)
            int meshed = -1;             // frame whose design `current` holds
            DesignMesh current;
        };

        void Normalize(TriMesh& m, float size) {
            Vector3 lo = { FLT_MAX, FLT_MAX, FLT_MAX }, hi = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
            for (Vector3 p : m.positions) { lo = Vector3Min(lo, p); hi = Vector3Max(hi, p); }
            Vector3 c = Vector3Scale(Vector3Add(lo, hi), 0.5f);
            Vector3 e = Vector3Subtract(hi, lo);
            float big = fmaxf(e.x, fmaxf(e.y, e.z));
            float k = size > 0.0f && big > 0.0f ? size / big : 1.0f;
            for (Vector3& p : m.positions) p = Vector3Scale(Vector3Subtract(p, c), k);
            m.box = { Vector3Scale(Vector3Subtract(lo, c), k), Vector3Scale(Vector3Subtract(hi, c), k) };
        }

        void BodyCorners(const Design& d, const std::vector<Rigid>& xf, std::vector<Vector3>* out) {
            for (int g : d.Bodies()) {
                if (d.GroupShapes(d.groups[g].id).empty()) continue;
                BoundingBox b = d.BodyBounds(d.groups[g].id);
                for (int k = 0; k < 8; k++)
                    out->push_back(xf[g].Apply({ (k & 1) ? b.max.x : b.min.x, (k & 2) ? b.max.y : b.min.y, (k & 4) ? b.max.z : b.min.z }));
            }
        }

        std::string Pct(float v) { return Utils::Format("%.0f%%", v * 100.0f); }
    }

    // ---- public --------------------------------------------------------------------------------------------------

    std::vector<std::string> SuiteMeshes(const std::string& folder) {
        std::vector<std::string> out;
        std::error_code ec;
        if (folder.empty() || !fs::is_directory(folder, ec)) return out;
        for (const fs::directory_entry& e : fs::directory_iterator(folder, ec)) {
            if (!e.is_regular_file(ec)) continue;
            std::string ext = e.path().extension().string();
            for (char& ch : ext) ch = (char)tolower((unsigned char)ch);
            if (ext == ".obj") out.push_back(e.path().string());
        }
        std::sort(out.begin(), out.end());
        return out;
    }

    // Positions and faces only (polygons are fanned; "v/vt/vn" and negative indices are fine).
    bool LoadObj(const std::string& path, TriMesh* out, std::string* error) {
        std::ifstream in(path);
        if (!in) { *error = "can't open it"; return false; }
        TriMesh m;
        std::string line;
        std::vector<long> face;
        while (std::getline(in, line)) {
            if (line.size() < 2) continue;
            if (line[0] == 'v' && isspace((unsigned char)line[1])) {
                Vector3 p = { 0, 0, 0 };
                if (sscanf(line.c_str() + 2, "%f %f %f", &p.x, &p.y, &p.z) == 3) m.positions.push_back(p);
            } else if (line[0] == 'f' && isspace((unsigned char)line[1])) {
                face.clear();
                std::istringstream ls(line.substr(2));
                std::string tok;
                while (ls >> tok) {
                    long i = strtol(tok.c_str(), nullptr, 10);
                    if (i < 0) i = (long)m.positions.size() + i + 1;
                    face.push_back(i - 1);
                }
                for (size_t k = 2; k < face.size(); k++) {
                    long a = face[0], b = face[k - 1], c = face[k];
                    long n = (long)m.positions.size();
                    if (a < 0 || b < 0 || c < 0 || a >= n || b >= n || c >= n) continue;
                    m.indices.insert(m.indices.end(), { (unsigned)a, (unsigned)b, (unsigned)c });
                }
            }
        }
        if (m.indices.empty()) { *error = "no faces"; return false; }
        *out = std::move(m);
        return true;
    }

    bool FfmpegAvailable() {
        return std::system("ffmpeg -version >" SUITE_NULL " 2>&1") == 0;
    }

    SuiteResult RunSuite(const SuiteParams& p, JobControl& c, std::shared_ptr<SuiteLive> live) {
        SuiteResult res;
        auto setPhase = [&](const std::string& phase, float progress) {
            c.SetStage(phase);
            if (!live) return;
            std::lock_guard<std::mutex> l(live->mutex);
            live->phase = phase;
            live->progress = progress;
        };
        std::vector<std::string> files = SuiteMeshes(p.folder);
        if (files.empty()) { res.message = "No .obj files in " + p.folder + "."; return res; }
        if (!FfmpegAvailable()) {
            res.message = "The test suite needs ffmpeg on the PATH to write the video (Arch: pacman -S ffmpeg).";
            return res;
        }
        fs::path outDir = fs::path(p.folder) / "autodesign_suite";
        std::error_code ec;
        fs::create_directories(outDir, ec);
        if (ec) { res.message = "Can't create " + outDir.string() + ": " + ec.message(); return res; }
        res.folder = outDir.string();

        const int fps = std::clamp(p.fps, 1, 120);
        const int searchFrames = std::max(2, (int)lroundf(p.searchSeconds * fps));
        const int motionFrames = std::max(2, (int)lroundf(p.motionSeconds * fps));
        const int holdFrames = std::max(0, (int)lroundf(p.holdSeconds * fps));

        // 1. the searches, one mesh at a time (each uses every core)
        std::vector<Entry> entries;
        std::vector<std::string> skipped;
        for (size_t i = 0; i < files.size(); i++) {
            Entry e;
            e.name = fs::path(files[i]).stem().string();
            if (live) {
                std::lock_guard<std::mutex> l(live->mutex);
                live->index = (int)i;
                live->count = (int)files.size();
                live->name = e.name;
            }
            setPhase("Searching", 0.0f);
            std::string err;
            if (!LoadObj(files[i], &e.mesh, &err)) {
                skipped.push_back(e.name + " (" + err + ")");
                continue;
            }
            Normalize(e.mesh, p.meshSize);
            AutoParams ap = p.generate;
            ap.fromCurrent = false;
            auto search = std::make_shared<AutoLive>();
            search->recordFrames = searchFrames;
            if (live) {
                std::lock_guard<std::mutex> l(live->mutex);
                live->search = search;
            }
            AutoResult r = AutoDesign(e.mesh, Design{}, ap, c, search);
            if (c.Cancelled()) { res.cancelled = true; res.message = "Test suite stopped."; return res; }
            if (!r.ok) {
                skipped.push_back(e.name + " (" + r.message + ")");
                continue;
            }
            e.result = r.design;
            e.score = r.score;
            e.iterations = ap.iterations;
            {
                std::lock_guard<std::mutex> l(search->mutex);
                e.frames = search->frames;
            }
            if (e.frames.empty()) e.frames.push_back({ r.design, r.score, ap.iterations });
            std::ofstream(outDir / (e.name + ".txt")) << r.design.Serialize();
            entries.push_back(std::move(e));
        }
        if (live) {
            std::lock_guard<std::mutex> l(live->mutex);
            live->search.reset();
        }
        if (entries.empty()) { res.message = "No mesh could be used: " + (skipped.empty() ? std::string() : skipped[0]); return res; }

        // 2. layout: a square grid of boxes, each a label row, the picture and a status area
        const int S = std::max(256, p.videoSize) & ~1;
        const int n = (int)entries.size();
        const int cols = (int)ceilf(sqrtf((float)n)), rows = (n + cols - 1) / cols;
        const int gap = std::max(8, S / 90);
        const int cell = (S - gap * (cols + 1)) / cols;
        const int top = (S - rows * cell - (rows - 1) * gap) / 2;
        const int ts = cell >= 520 ? 3 : (cell >= 200 ? 2 : 1);  // text scale
        const int pad = 4 + 3 * ts;
        const int titleH = 10 * ts + 2 * pad;
        const int statusLines = 3;
        const int statusH = statusLines * (10 * ts) + (statusLines - 1) * 2 * ts + 2 * pad + 4;
        const int picW = cell - 2 * pad, picH = std::max(16, cell - titleH - statusH);
        const int ss = 2;    // supersampling
        const int rw = picW * ss, rh = picH * ss;
        const float voxelShare = 1.0f / 90.0f;

        // a fixed camera per mesh: everything it will show, from the search to the motion, stays in the picture
        setPhase("Framing", 0.0f);
        ParallelFor(n, Threads(), [&](int i) {
            Entry& e = entries[i];
            std::vector<Vector3> pts;
            for (int k = 0; k < 8; k++)
                pts.push_back({ (k & 1) ? e.mesh.box.max.x : e.mesh.box.min.x, (k & 2) ? e.mesh.box.max.y : e.mesh.box.min.y,
                                (k & 4) ? e.mesh.box.max.z : e.mesh.box.min.z });
            // the mesh and the result's whole motion (search snapshots stay near the mesh; box corners of every one
            // of them would frame far too loosely)
            for (int k = 0; k <= 8; k++) {
                BodyCorners(e.result, e.result.Transforms(k / 8.0f, true), &pts);
                BodyCorners(e.result, e.result.Transforms(k / 8.0f, true, Motion::Opening), &pts);
            }
            e.view = MakeView(pts, rw, rh);
            // the input mesh as a faint silhouette with an outline
            std::vector<unsigned char> mask((size_t)rw * rh, 0);
            Raster r;
            r.w = rw;
            r.h = rh;
            r.Clear(std::vector<Rgb>((size_t)rw * rh, White));
            std::vector<Vector3> sp(e.mesh.positions.size());
            for (size_t k = 0; k < sp.size(); k++) sp[k] = e.view.Project(e.mesh.positions[k]);
            for (size_t k = 0; k + 2 < e.mesh.indices.size(); k += 3)
                DrawTriangle(r, sp[e.mesh.indices[k]], sp[e.mesh.indices[k + 1]], sp[e.mesh.indices[k + 2]], Rgb{ 226, 231, 240 },
                             Rgb{ 226, 231, 240 }, Rgb{ 226, 231, 240 }, 1);
            e.ghost = r.color;
            for (int y = 0; y + 1 < rh; y++)
                for (int x = 0; x + 1 < rw; x++) {
                    size_t k = (size_t)y * rw + x;
                    if (r.id[k] != r.id[k + 1] || r.id[k] != r.id[k + rw]) e.ghost[k] = { 190, 198, 212 };
                }
        });

        // 3. the video: frames go straight into ffmpeg
        fs::path video = outDir / "suite.mp4";
        std::string vpath = video.string();
        if (vpath.find('"') != std::string::npos) { res.message = "The folder's path has a quote in it: ffmpeg can't be given it."; return res; }
        std::string cmd = Utils::Format("ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgb24 -s %dx%d -r %d -i - -an -c:v libx264 "
                                        "-preset veryfast -crf 18 -pix_fmt yuv420p -movflags +faststart \"%s\"", S, S, fps, vpath.c_str());
#ifndef _WIN32
        void (*oldPipe)(int) = signal(SIGPIPE, SIG_IGN);     // a dying ffmpeg must not take the editor with it
        FILE* pipe = SUITE_POPEN(cmd.c_str(), "w");
#else
        FILE* pipe = SUITE_POPEN(cmd.c_str(), "wb");
#endif
        if (!pipe) {
            res.message = "Could not start ffmpeg.";
            return res;
        }

        enum class Part { Search, Result, Closing, Closed, Opening, Open };
        struct Moment { Part part; int k; int of; };
        std::vector<Moment> timeline;
        for (int k = 0; k < searchFrames; k++) timeline.push_back({ Part::Search, k, searchFrames });
        for (int k = 0; k < holdFrames; k++) timeline.push_back({ Part::Result, k, holdFrames });
        for (int k = 0; k < motionFrames; k++) timeline.push_back({ Part::Closing, k, motionFrames });
        for (int k = 0; k < holdFrames; k++) timeline.push_back({ Part::Closed, k, holdFrames });
        for (int k = 0; k < motionFrames; k++) timeline.push_back({ Part::Opening, k, motionFrames });
        for (int k = 0; k < holdFrames * 2; k++) timeline.push_back({ Part::Open, k, holdFrames * 2 });

        Canvas cv;
        cv.Resize(S, S);
        std::vector<unsigned char> bytes((size_t)S * S * 3);
        std::vector<Raster> rasters(n);
        bool writeFailed = false;
        setPhase("Rendering", 0.0f);
        for (size_t f = 0; f < timeline.size() && !writeFailed; f++) {
            if (c.Cancelled()) break;
            const Moment& m = timeline[f];
            cv.Fill(0, 0, S, S, White);
            ParallelFor(n, Threads(), [&](int i) {
                Entry& e = entries[i];
                int frame = (int)e.frames.size() - 1;
                const Design* d = &e.result;
                float t = 0.0f;
                Motion motion = Motion::Closing;
                float ghost = 1.0f;
                if (m.part == Part::Search) {
                    frame = std::min((int)e.frames.size() - 1, m.k * (int)e.frames.size() / std::max(1, m.of));
                    d = &e.frames[frame].design;
                } else if (m.part == Part::Closing) {
                    t = (m.k + 1.0f) / m.of;
                    ghost = 1.0f - t;
                } else if (m.part == Part::Closed) {
                    t = 1.0f;
                    ghost = 0.0f;
                } else if (m.part == Part::Opening) {
                    t = 1.0f - (m.k + 1.0f) / m.of;
                    motion = Motion::Opening;
                    ghost = 1.0f - t;
                }
                int key = m.part == Part::Search ? frame : (int)e.frames.size();
                if (e.meshed != key) {
                    // a new best design (most frames repeat the one before)
                    bool same = m.part != Part::Search ? false : (e.meshed >= 0 && e.meshed < (int)e.frames.size() && e.frames[e.meshed].design == *d);
                    if (!same) e.current = MeshDesign(*d, Vector3Distance(e.mesh.box.min, e.mesh.box.max) * voxelShare);
                    e.meshed = key;
                }
                Raster& r = rasters[i];
                r.w = rw;
                r.h = rh;
                if (ghost >= 1.0f) {
                    r.Clear(e.ghost);
                } else {
                    std::vector<Rgb> bg(e.ghost.size());
                    for (size_t k = 0; k < bg.size(); k++) bg[k] = Mix(White, e.ghost[k], ghost);
                    r.Clear(bg);
                }
                DrawDesign(r, e.current, d->Transforms(t, true, motion), e.view);
            });

            // compose: boxes, pictures (downsampled), labels
            for (int i = 0; i < n; i++) {
                const Entry& e = entries[i];
                int col = i % cols, row = i / cols;
                int x0 = gap + col * (cell + gap), y0 = top + row * (cell + gap);
                cv.Frame(x0, y0, x0 + cell, y0 + cell, std::max(1, ts), Border);
                int px0 = x0 + pad, py0 = y0 + titleH;
                const Raster& r = rasters[i];
                for (int y = 0; y < picH; y++)
                    for (int x = 0; x < picW; x++) {
                        int sr = 0, sg = 0, sb = 0;
                        for (int k = 0; k < ss * ss; k++) {
                            const Rgb& q = r.color[(size_t)(y * ss + k / ss) * rw + x * ss + k % ss];
                            sr += q.r; sg += q.g; sb += q.b;
                        }
                        int q = ss * ss;
                        cv.px[(size_t)(py0 + y) * S + px0 + x] = { (unsigned char)(sr / q), (unsigned char)(sg / q), (unsigned char)(sb / q) };
                    }
                // label row: the mesh's name, and what is showing
                int searchFrame = std::min((int)e.frames.size() - 1, m.part == Part::Search ? m.k * (int)e.frames.size() / std::max(1, m.of) : (int)e.frames.size() - 1);
                const AutoFrame& af = e.frames[searchFrame];
                std::string state;
                switch (m.part) {
                    case Part::Search: state = Utils::Format("search %d%%", (int)lroundf(100.0f * af.iteration / std::max(1, e.iterations))); break;
                    case Part::Result: state = "result"; break;
                    case Part::Closing: state = "closing"; break;
                    case Part::Closed: state = "closed"; break;
                    case Part::Opening: state = "opening"; break;
                    case Part::Open: state = "open"; break;
                }
                int stateW = TextWidth(state, ts);
                DrawText(cv, x0 + pad, y0 + pad, ts, Fit(e.name, cell - 3 * pad - stateW, ts), Ink);
                DrawText(cv, x0 + cell - pad - stateW, y0 + pad, ts, state, Muted);
                // status: the search's progress, then the score of the design showing, flowed over two lines
                int by = y0 + titleH + picH + pad;
                float progress = m.part == Part::Search ? (float)af.iteration / std::max(1, e.iterations) : 1.0f;
                cv.Fill(x0 + pad, by, x0 + cell - pad, by + 4, Rgb{ 232, 235, 241 });
                cv.Fill(x0 + pad, by, x0 + pad + (int)((cell - 2 * pad) * std::clamp(progress, 0.0f, 1.0f)), by + 4, Accent);
                const AutoScore& s = af.score;
                std::vector<std::string> words = { Utils::Format("score %.3f", s.total), "cover " + Pct(s.fitCover),
                                                   "outside " + Pct(s.fitOutside), "sphere " + Pct(s.sphereIoU),
                                                   Utils::Format("overlap %.1f%%", (s.clashOpen + s.clashClosed) * 50.0f) };
                int ly = by + 4 + pad, maxW = cell - 2 * pad;
                std::string lineText;
                int lines = 0;
                for (size_t k = 0; k <= words.size() && lines < statusLines; k++) {
                    std::string next = k < words.size() ? (lineText.empty() ? words[k] : lineText + "  " + words[k]) : std::string();
                    if (k < words.size() && TextWidth(next, ts) <= maxW) { lineText = next; continue; }
                    if (!lineText.empty()) {
                        DrawText(cv, x0 + pad, ly, ts, Fit(lineText, maxW, ts), Muted);
                        ly += 10 * ts + 2 * ts;
                        lines++;
                    }
                    lineText = k < words.size() ? words[k] : std::string();
                }
            }
            for (size_t k = 0; k < cv.px.size(); k++) {
                bytes[k * 3] = cv.px[k].r;
                bytes[k * 3 + 1] = cv.px[k].g;
                bytes[k * 3 + 2] = cv.px[k].b;
            }
            if (fwrite(bytes.data(), 1, bytes.size(), pipe) != bytes.size()) writeFailed = true;
            float prog = (float)(f + 1) / timeline.size();
            c.SetProgress(prog);
            if (live) {
                std::lock_guard<std::mutex> l(live->mutex);
                live->progress = prog;
            }
        }
        int status = SUITE_PCLOSE(pipe);
#ifndef _WIN32
        signal(SIGPIPE, oldPipe);
#endif
        if (c.Cancelled()) {
            fs::remove(video, ec);
            res.cancelled = true;
            res.message = "Test suite stopped (the designs found so far are in " + res.folder + ").";
            return res;
        }
        if (writeFailed || status != 0) {
            res.message = "ffmpeg failed while writing " + vpath + " (is libx264 available?).";
            return res;
        }
        res.ok = true;
        res.video = vpath;
        res.message = Utils::Format("Test suite done: %d mesh%s, %.0f s of video in %s (designs saved next to it).", n,
                                    n == 1 ? "" : "es", timeline.size() / (float)fps, vpath.c_str());
        if (!skipped.empty()) {
            res.message += " Skipped:";
            for (const std::string& s : skipped) res.message += " " + s;
        }
        return res;
    }
}
