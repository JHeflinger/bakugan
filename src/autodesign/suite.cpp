#include "suite.hpp"
#include "common.hpp"
#include "suite_draw.hpp"
#include "util/utils.hpp"
#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
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

// The test suite, in four steps (RunSuite):
//   1. find the .obj files, check ffmpeg, make the output folder
//   2. run the generator on each mesh, recording the best design at evenly spaced moments
//   3. frame each mesh: a fixed camera that keeps everything it will show in view, and the mesh as a ghost behind it
//   4. render the video's timeline (the search, a pause, closing, closed, opening, open) straight into ffmpeg
namespace Bakugan {
    namespace fs = std::filesystem;
    using namespace Suite;

    namespace {
        // One mesh's run and how it is drawn.
        struct Entry {
            std::string name;
            TriMesh mesh;
            std::vector<AutoFrame> frames;   // the best design over the search
            Design result;
            int iterations = 0;
            View view;
            std::vector<Rgb> ghost;          // the input mesh, faint (picture resolution)
            int meshed = -1;                 // which frame `current` holds (frames.size(): the result)
            DesignMesh current;
        };

        // Where the boxes go: a square grid, each a label row, the picture and three status lines.
        struct Layout {
            int S = 0, cols = 1, rows = 1, gap = 8, cell = 0, top = 0;
            int ts = 1, pad = 7, titleH = 0, statusH = 0, picW = 0, picH = 0;
            static constexpr int statusLines = 3, ss = 2;   // status lines; supersampling of the pictures
            int RasterW() const { return picW * ss; }
            int RasterH() const { return picH * ss; }
            Layout(int videoSize, int n) {
                S = std::max(256, videoSize) & ~1;
                cols = (int)ceilf(sqrtf((float)n));
                rows = (n + cols - 1) / cols;
                gap = std::max(8, S / 90);
                cell = (S - gap * (cols + 1)) / cols;
                top = (S - rows * cell - (rows - 1) * gap) / 2;
                ts = cell >= 520 ? 3 : (cell >= 200 ? 2 : 1);
                pad = 4 + 3 * ts;
                titleH = 10 * ts + 2 * pad;
                statusH = statusLines * (10 * ts) + (statusLines - 1) * 2 * ts + 2 * pad + 4;
                picW = cell - 2 * pad;
                picH = std::max(16, cell - titleH - statusH);
            }
        };

        // The video, moment by moment.
        enum class Part { Search, Result, Closing, Closed, Opening, Open };
        struct Moment {
            Part part;
            int k, of;                       // k-th frame of `of` in this part
        };
        std::vector<Moment> Timeline(int search, int motion, int hold) {
            std::vector<Moment> out;
            auto add = [&](Part p, int n) {
                for (int k = 0; k < n; k++) out.push_back({ p, k, n });
            };
            add(Part::Search, search);
            add(Part::Result, hold);
            add(Part::Closing, motion);
            add(Part::Closed, hold);
            add(Part::Opening, motion);
            add(Part::Open, hold * 2);
            return out;
        }
        const char* PartName(Part p) {
            switch (p) {
                case Part::Search: return "search";
                case Part::Result: return "result";
                case Part::Closing: return "closing";
                case Part::Closed: return "closed";
                case Part::Opening: return "opening";
                case Part::Open: return "open";
            }
            return "";
        }
        int SearchFrame(const Entry& e, const Moment& m) {
            int last = (int)e.frames.size() - 1;
            return m.part == Part::Search ? std::min(last, m.k * (int)e.frames.size() / std::max(1, m.of)) : last;
        }

        // Centred, its longest side `size` mm (0: as it is).
        void Normalize(TriMesh& m, float size) {
            Vector3 lo = { FLT_MAX, FLT_MAX, FLT_MAX }, hi = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
            for (Vector3 p : m.positions) { lo = Vector3Min(lo, p); hi = Vector3Max(hi, p); }
            Vector3 c = Vector3Scale(Vector3Add(lo, hi), 0.5f);
            float big = Auto::MaxOf(Vector3Subtract(hi, lo));
            float k = size > 0.0f && big > 0.0f ? size / big : 1.0f;
            for (Vector3& p : m.positions) p = Vector3Scale(Vector3Subtract(p, c), k);
            m.box = { Vector3Scale(Vector3Subtract(lo, c), k), Vector3Scale(Vector3Subtract(hi, c), k) };
        }

        void BodyCorners(const Design& d, const std::vector<Rigid>& xf, std::vector<Vector3>* out) {
            for (int g : d.Bodies()) {
                if (d.GroupShapes(d.groups[g].id).empty()) continue;
                BoundingBox b = d.BodyBounds(d.groups[g].id);
                for (int k = 0; k < 8; k++) out->push_back(xf[g].Apply(BoxCorner(b, k)));
            }
        }

        std::string Pct(float v) { return Utils::Format("%.0f%%", v * 100.0f); }

        // A design's score in a few words.
        std::vector<std::string> ScoreWords(const AutoScore& s) {
            std::vector<std::string> w = { Utils::Format("score %.3f", s.total) };
            if (s.silhouettes) w.insert(w.end(), { "outline " + Pct(s.viewCover), "extra " + Pct(s.viewExtra), "sphere " + Pct(s.sphereViewCover) });
            else w.insert(w.end(), { "cover " + Pct(s.fitCover), "outside " + Pct(s.fitOutside), "sphere " + Pct(s.sphereIoU) });
            w.push_back(Utils::Format("overlap %.1f%%", (s.clashOpen + s.clashClosed) * 50.0f));
            w.push_back(Utils::Format("moving %.1f%%", s.clashMotion * 100.0f));
            return w;
        }

        // ffmpeg reading raw RGB frames on its standard input, writing an H.264 mp4.
        class Encoder {
        public:
            bool Open(const std::string& path, int size, int fps) {
                std::string cmd = Utils::Format("ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgb24 -s %dx%d -r %d -i - -an -c:v libx264 "
                                                "-preset veryfast -crf 18 -pix_fmt yuv420p -movflags +faststart \"%s\"", size, size, fps, path.c_str());
#ifndef _WIN32
                m_OldPipe = signal(SIGPIPE, SIG_IGN);     // a dying ffmpeg must not take the editor with it
                m_Pipe = SUITE_POPEN(cmd.c_str(), "w");
#else
                m_Pipe = SUITE_POPEN(cmd.c_str(), "wb");
#endif
                return m_Pipe != nullptr;
            }
            bool Write(const std::vector<unsigned char>& rgb) { return fwrite(rgb.data(), 1, rgb.size(), m_Pipe) == rgb.size(); }
            // True when ffmpeg finished cleanly.
            bool Close() {
                int status = m_Pipe ? SUITE_PCLOSE(m_Pipe) : -1;
                m_Pipe = nullptr;
#ifndef _WIN32
                signal(SIGPIPE, m_OldPipe);
#endif
                return status == 0;
            }
            ~Encoder() { if (m_Pipe) Close(); }
        private:
            FILE* m_Pipe = nullptr;
#ifndef _WIN32
            void (*m_OldPipe)(int) = SIG_DFL;
#endif
        };

        // Step 3 for one mesh: everything it will show, from the search to the motion, stays in the picture (search
        // snapshots stay near the mesh, so the mesh's box and the result's whole motion frame it).
        void Frame(Entry& e, int w, int h) {
            std::vector<Vector3> pts;
            for (int k = 0; k < 8; k++) pts.push_back(BoxCorner(e.mesh.box, k));
            for (int k = 0; k <= 8; k++) {
                BodyCorners(e.result, e.result.Transforms(k / 8.0f, true), &pts);
                BodyCorners(e.result, e.result.Transforms(k / 8.0f, true, Motion::Opening), &pts);
            }
            e.view = View::Framing(pts, w, h);
            e.ghost = Ghost(e.mesh, e.view, w, h);
        }

        // One mesh's picture at one moment: the design showing then, in its pose, over the ghost (which fades while
        // the design is closed).
        void Render(Entry& e, const Moment& m, int w, int h, Raster& r) {
            int frame = SearchFrame(e, m);
            const Design* d = m.part == Part::Search ? &e.frames[frame].design : &e.result;
            float t = 0.0f, ghost = 1.0f;
            Motion motion = Motion::Closing;
            if (m.part == Part::Closing) {
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
            // mesh the design only when it changes (most frames repeat the one before)
            int key = m.part == Part::Search ? frame : (int)e.frames.size();
            if (e.meshed != key) {
                bool same = m.part == Part::Search && e.meshed >= 0 && e.meshed < (int)e.frames.size() && e.frames[e.meshed].design == *d;
                if (!same) e.current = MeshDesign(*d, Vector3Distance(e.mesh.box.min, e.mesh.box.max) / 90.0f);
                e.meshed = key;
            }
            if (ghost >= 1.0f) {
                r.Clear(w, h, e.ghost);
            } else {
                std::vector<Rgb> bg(e.ghost.size());
                for (size_t k = 0; k < bg.size(); k++) bg[k] = Mix(White, e.ghost[k], ghost);
                r.Clear(w, h, bg);
            }
            DrawDesign(r, e.current, d->Transforms(t, true, motion), e.view);
        }

        // One mesh's box in the video: border, picture (downsampled), name and moment, progress bar, score.
        void DrawCell(Canvas& cv, const Layout& L, int i, const Entry& e, const Raster& r, const Moment& m) {
            int x0 = L.gap + (i % L.cols) * (L.cell + L.gap), y0 = L.top + (i / L.cols) * (L.cell + L.gap);
            cv.Frame(x0, y0, x0 + L.cell, y0 + L.cell, std::max(1, L.ts), Border);
            int px0 = x0 + L.pad, py0 = y0 + L.titleH, ss = Layout::ss, rw = L.RasterW();
            for (int y = 0; y < L.picH; y++)
                for (int x = 0; x < L.picW; x++) {
                    int sr = 0, sg = 0, sb = 0;
                    for (int k = 0; k < ss * ss; k++) {
                        const Rgb& q = r.color[(size_t)(y * ss + k / ss) * rw + x * ss + k % ss];
                        sr += q.r; sg += q.g; sb += q.b;
                    }
                    int q = ss * ss;
                    cv.px[(size_t)(py0 + y) * cv.w + px0 + x] = { (unsigned char)(sr / q), (unsigned char)(sg / q), (unsigned char)(sb / q) };
                }
            const AutoFrame& af = e.frames[SearchFrame(e, m)];
            float progress = m.part == Part::Search ? (float)af.iteration / std::max(1, e.iterations) : 1.0f;
            std::string state = m.part == Part::Search ? Utils::Format("search %d%%", (int)lroundf(100.0f * progress)) : PartName(m.part);
            int stateW = TextWidth(state, L.ts);
            cv.Text(x0 + L.pad, y0 + L.pad, L.ts, FitText(e.name, L.cell - 3 * L.pad - stateW, L.ts), Ink);
            cv.Text(x0 + L.cell - L.pad - stateW, y0 + L.pad, L.ts, state, Muted);
            int by = y0 + L.titleH + L.picH + L.pad;
            cv.Fill(x0 + L.pad, by, x0 + L.cell - L.pad, by + 4, Track);
            cv.Fill(x0 + L.pad, by, x0 + L.pad + (int)((L.cell - 2 * L.pad) * std::clamp(progress, 0.0f, 1.0f)), by + 4, Accent);
            // the score's words, flowed over the status lines
            std::vector<std::string> words = ScoreWords(af.score);
            int ly = by + 4 + L.pad, maxW = L.cell - 2 * L.pad, lines = 0;
            std::string line;
            for (size_t k = 0; k <= words.size() && lines < Layout::statusLines; k++) {
                std::string next = k < words.size() ? (line.empty() ? words[k] : line + "  " + words[k]) : std::string();
                if (k < words.size() && TextWidth(next, L.ts) <= maxW) { line = next; continue; }
                if (!line.empty()) {
                    cv.Text(x0 + L.pad, ly, L.ts, FitText(line, maxW, L.ts), Muted);
                    ly += 12 * L.ts;
                    lines++;
                }
                line = k < words.size() ? words[k] : std::string();
            }
        }
    }

    // ---- public ---------------------------------------------------------------------------------------------------------

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
                long n = (long)m.positions.size();
                for (size_t k = 2; k < face.size(); k++) {
                    long a = face[0], b = face[k - 1], c = face[k];
                    if (a < 0 || b < 0 || c < 0 || a >= n || b >= n || c >= n) continue;
                    m.indices.insert(m.indices.end(), { (unsigned)a, (unsigned)b, (unsigned)c });
                }
            }
        }
        if (m.indices.empty()) { *error = "no faces"; return false; }
        *out = std::move(m);
        return true;
    }

    bool FfmpegAvailable() { return std::system("ffmpeg -version >" SUITE_NULL " 2>&1") == 0; }

    SuiteResult RunSuite(const SuiteParams& p, JobControl& c, std::shared_ptr<SuiteLive> live) {
        SuiteResult res;
        auto show = [&](auto&& update) {
            if (!live) return;
            std::lock_guard<std::mutex> l(live->mutex);
            update(*live);
        };
        auto phase = [&](const std::string& name) {
            c.SetStage(name);
            show([&](SuiteLive& s) { s.phase = name; s.progress = 0.0f; });
        };

        // 1. the meshes, ffmpeg, the output folder
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
        fs::path video = outDir / "suite.mp4";
        std::string vpath = video.string();
        if (vpath.find('"') != std::string::npos) { res.message = "The folder's path has a quote in it: ffmpeg can't be given it."; return res; }

        const int fps = std::clamp(p.fps, 1, 120);
        auto frames = [&](float seconds, int least) { return std::max(least, (int)lroundf(seconds * fps)); };
        const std::vector<Moment> timeline = Timeline(frames(p.searchSeconds, 2), frames(p.motionSeconds, 2), frames(p.holdSeconds, 0));
        const int searchFrames = timeline.empty() ? 0 : timeline[0].of;

        // 2. the searches, one mesh at a time (each uses every core)
        std::vector<Entry> entries;
        std::vector<std::string> skipped;
        for (size_t i = 0; i < files.size(); i++) {
            Entry e;
            e.name = fs::path(files[i]).stem().string();
            show([&](SuiteLive& s) { s.index = (int)i; s.count = (int)files.size(); s.name = e.name; });
            phase("Searching");
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
            show([&](SuiteLive& s) { s.search = search; });
            AutoResult r = AutoDesign(e.mesh, Design{}, ap, c, search);
            if (c.Cancelled()) { res.cancelled = true; res.message = "Test suite stopped."; return res; }
            if (!r.ok) {
                skipped.push_back(e.name + " (" + r.message + ")");
                continue;
            }
            e.result = r.design;
            e.iterations = ap.iterations;
            {
                std::lock_guard<std::mutex> l(search->mutex);
                e.frames = search->frames;
            }
            if (e.frames.empty()) e.frames.push_back({ r.design, r.score, ap.iterations });
            std::ofstream(outDir / (e.name + ".txt")) << r.design.Serialize();
            entries.push_back(std::move(e));
        }
        show([&](SuiteLive& s) { s.search.reset(); });
        if (entries.empty()) { res.message = "No mesh could be used: " + (skipped.empty() ? std::string() : skipped[0]); return res; }

        // 3. a fixed camera per mesh
        const int n = (int)entries.size();
        const Layout L(p.videoSize, n);
        const int rw = L.RasterW(), rh = L.RasterH();
        Auto::Pool pool(std::max(0, (int)std::thread::hardware_concurrency() - 1));
        phase("Framing");
        pool.For(n, [&](int i) { Frame(entries[i], rw, rh); });

        // 4. the video, frame by frame into ffmpeg
        Encoder encoder;
        if (!encoder.Open(vpath, L.S, fps)) { res.message = "Could not start ffmpeg."; return res; }
        Canvas cv;
        cv.Resize(L.S, L.S);
        std::vector<unsigned char> bytes;
        std::vector<Raster> rasters(n);
        bool writeFailed = false;
        phase("Rendering");
        for (size_t f = 0; f < timeline.size() && !writeFailed && !c.Cancelled(); f++) {
            const Moment& m = timeline[f];
            pool.For(n, [&](int i) { Render(entries[i], m, rw, rh, rasters[i]); });
            cv.Fill(0, 0, L.S, L.S, White);
            for (int i = 0; i < n; i++) DrawCell(cv, L, i, entries[i], rasters[i], m);
            cv.Rgb24(&bytes);
            writeFailed = !encoder.Write(bytes);
            float progress = (float)(f + 1) / timeline.size();
            c.SetProgress(progress);
            show([&](SuiteLive& s) { s.progress = progress; });
        }
        bool encoded = encoder.Close();
        if (c.Cancelled()) {
            fs::remove(video, ec);
            res.cancelled = true;
            res.message = "Test suite stopped (the designs found so far are in " + res.folder + ").";
            return res;
        }
        if (writeFailed || !encoded) {
            res.message = "ffmpeg failed while writing " + vpath + " (is libx264 available?).";
            return res;
        }
        res.ok = true;
        res.video = vpath;
        res.message = Utils::Format("Test suite done: %d mesh%s, %.0f s of video in %s (designs saved next to it).", n, n == 1 ? "" : "es",
                                    timeline.size() / (float)fps, vpath.c_str());
        if (!skipped.empty()) {
            res.message += " Skipped:";
            for (const std::string& s : skipped) res.message += " " + s;
        }
        return res;
    }
}
