// foldaball, on the command line: the generator and the test suite without the editor, for machines with no display or
// GPU. Nothing here opens a window or a GL context. See cli/README.md.
//
//   foldaball optimize mesh.obj -c settings.ini -o results/
//   foldaball suite meshes/ -c settings.ini
//   foldaball score design.txt mesh.obj
//   foldaball config > settings.ini
//
// TBB first: raylib's macros (PI, EPSILON, ...) must not leak into its headers.
#include <tbb/global_control.h>
#include "cli/config.hpp"
#include "util/utils.hpp"
#include "core/app.hpp"
#include <raymath.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <thread>

namespace fs = std::filesystem;
using namespace Bakugan;
using Clock = std::chrono::steady_clock;

namespace {
    // ---- stopping ---------------------------------------------------------------------------------------------------------
    // Ctrl+C or SIGTERM (a scheduler's time limit): the search stops at its next step and the best design so far is saved.
    // A second one quits at once.
    std::atomic<int> g_signals{ 0 };
    extern "C" void OnSignal(int) { g_signals.fetch_add(1); }

    // Watches a running job from another thread: passes a stop on to it, prints a progress line now and then, and calls
    // `tick` (checkpoints) several times a second.
    class Watcher {
    public:
        Watcher(JobControl& control, int everySeconds, std::function<std::string()> status, std::function<void()> tick = {})
            : m_Control(control), m_Every(everySeconds), m_Status(std::move(status)), m_Tick(std::move(tick)), m_Baseline(g_signals.load()) {
            m_Thread = std::thread([this] { Loop(); });
        }
        ~Watcher() {
            m_Done = true;
            m_Thread.join();
        }
    private:
        void Loop() {
            const auto start = Clock::now();
            auto last = start;
            bool stopping = false;
            while (!m_Done) {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                int signals = g_signals.load() - m_Baseline;
                if (signals >= 2) {
                    fprintf(stderr, "\nQuitting now.\n");
                    std::_Exit(130);
                }
                if (signals >= 1 && !stopping) {
                    stopping = true;
                    m_Control.Cancel();
                    fprintf(stderr, "\nStopping: the best so far will be saved (once more to quit at once)...\n");
                }
                if (m_Tick) m_Tick();
                auto now = Clock::now();
                if (m_Every > 0 && now - last >= std::chrono::seconds(m_Every) && m_Status) {
                    last = now;
                    double s = std::chrono::duration<double>(now - start).count();
                    fprintf(stderr, "[%s] %s\n", Duration(s).c_str(), m_Status().c_str());
                }
            }
        }
        JobControl& m_Control;
        int m_Every;
        std::function<std::string()> m_Status;
        std::function<void()> m_Tick;
        int m_Baseline;
        std::atomic<bool> m_Done{ false };
        std::thread m_Thread;
    public:
        static std::string Duration(double s) {
            int t = (int)s;
            return t >= 3600 ? Utils::Format("%d:%02d:%02d", t / 3600, t / 60 % 60, t % 60) : Utils::Format("%02d:%02d", t / 60, t % 60);
        }
    };

    // ---- arguments --------------------------------------------------------------------------------------------------------
    struct Args {
        std::string command;
        std::vector<std::string> positional;
        std::vector<std::string> configs;
        std::vector<std::pair<std::string, std::string>> sets;      // in the order given
        bool quiet = false, help = false;
    };

    const char* kUsage =
        "foldaball: carve a sphere into hinged pieces that open into a mesh, without the editor.\n"
        "\n"
        "usage:\n"
        "  foldaball optimize <mesh.obj> [options]     fit one mesh (what the Generate tab's button does)\n"
        "  foldaball suite <folder> [options]          every .obj in a folder: designs, pictures, summary.csv and a video\n"
        "  foldaball score <design.txt> <mesh.obj>     score a saved design against a mesh, as it is (no search)\n"
        "  foldaball config [options]                  print every setting, with its value and a comment (the settings file)\n"
        "\n"
        "options:\n"
        "  -c, --config FILE      settings file (INI; repeat it: later files override earlier ones)\n"
        "  -s, --set KEY=VALUE    one setting, e.g. -s optimizer.steps=3000 (repeatable; beats the files)\n"
        "  -o, --out DIR          where results go (output.dir)\n"
        "  -j, --threads N        worker threads (runtime.threads; default: every core)\n"
        "      --steps N          optimizer.steps\n"
        "      --seed N           optimizer.seed\n"
        "      --start FILE       optimize: carry the search on from a saved design (.txt) instead of the sphere (start.design)\n"
        "      --no-video         suite: skip the video (and so ffmpeg)\n"
        "  -q, --quiet            no progress lines\n"
        "  -h, --help             this text\n"
        "\n"
        "Settings not given keep the editor's defaults. Ctrl+C (or SIGTERM) stops a run and saves the best design so far.\n"
        "Exit status: 0 done, 1 failed, 2 bad usage or settings, 130 stopped early.\n";

    bool ParseArgs(int argc, char** argv, Args* a, std::string* error) {
        auto value = [&](int& i, const std::string& arg, std::string* out) {
            size_t eq = arg.find('=');
            if (arg.rfind("--", 0) == 0 && eq != std::string::npos) { *out = arg.substr(eq + 1); return true; }
            if (i + 1 >= argc) { *error = "missing a value after " + arg; return false; }
            *out = argv[++i];
            return true;
        };
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i], v;
            std::string flag = arg.rfind("--", 0) == 0 ? arg.substr(0, arg.find('=')) : arg;
            if (flag == "-h" || flag == "--help") a->help = true;
            else if (flag == "-q" || flag == "--quiet") a->quiet = true;
            else if (flag == "--no-video") a->sets.push_back({ "suite.video", "false" });
            else if (flag == "-c" || flag == "--config") { if (!value(i, arg, &v)) return false; a->configs.push_back(v); }
            else if (flag == "-s" || flag == "--set") {
                if (!value(i, arg, &v)) return false;
                size_t eq = v.find('=');
                if (eq == std::string::npos) { *error = "expected KEY=VALUE after " + flag + ", got '" + v + "'"; return false; }
                a->sets.push_back({ v.substr(0, eq), v.substr(eq + 1) });
            }
            else if (flag == "-o" || flag == "--out") { if (!value(i, arg, &v)) return false; a->sets.push_back({ "output.dir", v }); }
            else if (flag == "-j" || flag == "--threads") { if (!value(i, arg, &v)) return false; a->sets.push_back({ "runtime.threads", v }); }
            else if (flag == "--steps") { if (!value(i, arg, &v)) return false; a->sets.push_back({ "optimizer.steps", v }); }
            else if (flag == "--seed") { if (!value(i, arg, &v)) return false; a->sets.push_back({ "optimizer.seed", v }); }
            else if (flag == "--start") { if (!value(i, arg, &v)) return false; a->sets.push_back({ "start.design", v }); }
            else if (arg.size() > 1 && arg[0] == '-') { *error = "unknown option " + arg + " (see --help)"; return false; }
            else if (a->command.empty()) a->command = arg;
            else a->positional.push_back(arg);
        }
        return true;
    }

    // defaults, then the files, then the command line, then the positional arguments
    bool Configure(const Args& a, Cli::Config& cfg, std::string* error) {
        for (const std::string& f : a.configs)
            if (!cfg.LoadFile(f, error)) return false;
        for (const auto& kv : a.sets)
            if (!cfg.Set(kv.first, kv.second, error)) return false;
        if (a.quiet) cfg.runtime.progressSeconds = 0;
        return true;
    }

    // ---- small helpers ----------------------------------------------------------------------------------------------------
    bool Write(const fs::path& path, const std::string& text) {
        std::string err;
        if (Utils::WriteTextFile(path.string(), text, &err)) return true;
        fprintf(stderr, "warning: %s\n", err.c_str());
        return false;
    }

    bool MakeDir(const fs::path& dir, std::string* error) {
        std::error_code ec;
        fs::create_directories(dir, ec);
        if (ec) { *error = "can't create " + dir.string() + ": " + ec.message(); return false; }
        return true;
    }

    std::string Json(const std::string& s) {
        std::string o = "\"";
        for (char c : s) {
            if (c == '"' || c == '\\') { o += '\\'; o += c; }
            else if (c == '\n') o += "\\n";
            else if ((unsigned char)c < 0x20) o += ' ';
            else o += c;
        }
        return o + "\"";
    }

    bool LoadMesh(const std::string& path, float size, TriMesh* mesh, std::string* error) {
        if (path.empty()) { *error = "no mesh given (a .obj file)"; return false; }
        if (!Utils::ReadOBJ(path, mesh, error)) return false;
        Utils::ComputeNormals(*mesh);
        if (size > 0.0f) NormalizeMesh(*mesh, size);
        return true;
    }

    float LongestSide(const TriMesh& m) {
        Vector3 e = Vector3Subtract(m.box.max, m.box.min);
        return fmaxf(e.x, fmaxf(e.y, e.z));
    }

    // The mesh where the design was fitted to it.
    TriMesh Placed(const TriMesh& mesh, const Placement& t) {
        TriMesh m = mesh;
        for (Vector3& q : m.positions) q = t.Apply(q);
        m.box = { t.Apply(mesh.box.min), t.Apply(mesh.box.max) };
        return m;
    }

    // The settings that were used, next to the results: running it again with this file gives the same run.
    void SaveSettings(const Cli::Config& cfg, const fs::path& path) { Write(path, cfg.Dump()); }

    std::string ScoreLines(const AutoScore& s) {
        std::string t;
        t += Utils::Format("  loss                 %.4f\n", s.total);
        t += Utils::Format("  outline covered      %.1f%%   extra %.1f%%   worst view %.1f%%\n", s.viewCover * 100.0f, s.viewExtra * 100.0f, s.worstView * 100.0f);
        t += Utils::Format("  overlap, open        %.3f%% of the sphere   worst on the way %.3f%%\n", s.clashOpen * 100.0f, s.clashMotion * 100.0f);
        t += Utils::Format("  surface carved       %.1f%%\n", s.surfaceCarved * 100.0f);
        t += Utils::Format("  hinges off pieces    %.4f   (farthest a piece is from an axis: %.2f mm)\n", s.hinges, s.hingeGap);
        if (s.dependents > 0) t += Utils::Format("  held hinges          %d of %d have a holder in reach\n", s.holdable, s.dependents);
        if (s.depth > 0.0f) t += Utils::Format("  depth mismatch       %.4f\n", s.depth);
        if (s.semantic > 0.0f) t += Utils::Format("  DINOv3               %.4f   (worst view %.4f)\n", s.semantic, s.worstSemantic);
        t += Utils::Format("  mesh placed          scaled %.3fx, moved %.2f mm\n", s.target.scale, Vector3Length(s.target.offset));
        return t;
    }

    // A new, empty folder in `parent` for this run: <stem>_1, <stem>_2, ... the first that doesn't exist yet.
    // create_directory fails when the folder is there already, so two runs started together can't share one.
    bool MakeRunDir(const fs::path& parent, const std::string& stem, fs::path* out, std::string* error) {
        std::error_code ec;
        fs::create_directories(parent, ec);
        if (ec) { *error = "can't create " + parent.string() + ": " + ec.message(); return false; }
        for (int n = 1; n < 1000000; n++) {
            fs::path dir = parent / (stem + "_" + std::to_string(n));
            if (fs::create_directory(dir, ec)) { *out = dir; return true; }
            if (ec) { *error = "can't create " + dir.string() + ": " + ec.message(); return false; }
        }
        *error = "no free folder name for " + stem + " in " + parent.string();
        return false;
    }

    std::string ScoreJson(const AutoScore& s) {
        return Utils::Format("{\"loss\": %.6g, \"fit\": %.6g, \"cover\": %.6g, \"extra\": %.6g, \"worst_view\": %.6g, \"collision\": %.6g, "
                             "\"overlap_open\": %.6g, \"overlap_motion\": %.6g, \"surface_carved\": %.6g, \"hinges\": %.6g, "
                             "\"dependency\": %.6g, \"depth\": %.6g, \"semantic\": %.6g, \"mesh_scale\": %.6g, \"mesh_moved_mm\": %.6g}",
                             s.total, s.fit, s.viewCover, s.viewExtra, s.worstView, s.collision, s.clashOpen, s.clashMotion,
                             s.surfaceCarved, s.hinges, s.dependency, s.depth, s.semantic, s.target.scale, Vector3Length(s.target.offset));
    }

    // Nearest neighbour enlargement, so a 64 pixel view is something to look at.
    void Enlarge(const std::vector<unsigned char>& rgba, int w, int h, std::vector<unsigned char>* out, int* ow, int* oh) {
        int k = std::max(1, 512 / std::max(1, std::min(w, h)));
        *ow = w * k;
        *oh = h * k;
        out->resize((size_t)*ow * *oh * 4);
        for (int y = 0; y < *oh; y++)
            for (int x = 0; x < *ow; x++)
                std::copy_n(&rgba[((size_t)(y / k) * w + x / k) * 4], 4, &(*out)[((size_t)y * *ow + x) * 4]);
    }

    // ---- optimize ---------------------------------------------------------------------------------------------------------
    int CmdOptimize(Cli::Config& cfg, const Args& a) {
        if (a.positional.size() > 1) { fprintf(stderr, "optimize takes one mesh.\n"); return 2; }
        if (!a.positional.empty()) cfg.input.mesh = a.positional[0];
        std::string err;
        if (!cfg.Validate(&err)) { fprintf(stderr, "settings: %s\n", err.c_str()); return 2; }
        if (cfg.generate.steps < 1) {
            fprintf(stderr, "optimizer.steps is 0, which only scores a design that exists already: use `foldaball score`.\n");
            return 2;
        }
        if (cfg.input.mesh.empty()) { fprintf(stderr, "usage: foldaball optimize <mesh.obj> (or input.mesh in the settings)\n"); return 2; }
        // a design to carry on from: read and checked now, before anything is written
        std::shared_ptr<AutoStart> startFrom;
        if (!cfg.input.startDesign.empty()) {
            auto s = std::make_shared<AutoStart>();
            Carving k;
            std::string notes;
            if (!ReadAutoStart(cfg.input.startDesign, s.get(), &err) || !CarvingFromDesign(s->design, &k, &err, &notes)) {
                fprintf(stderr, "can't start from %s: %s\n", cfg.input.startDesign.c_str(), err.c_str());
                return 1;
            }
            startFrom = s;
            if (!a.quiet)
                printf("Starting from %s: %d pieces, %d joints, %d hollows%s.\n", cfg.input.startDesign.c_str(), (int)k.pieces.size(),
                       (int)k.joints.size(), k.Hollows(), s->hasTarget ? "" : " (no saved mesh placement: the mesh starts where it is)");
        }
        TriMesh mesh;
        if (!LoadMesh(cfg.input.mesh, cfg.input.meshSize, &mesh, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        const fs::path baseDir = cfg.output.dir.empty() ? fs::path("foldaball_out") : fs::path(cfg.output.dir);
        const std::string name = cfg.output.name.empty() ? fs::path(cfg.input.mesh).stem().string() : cfg.output.name;
        fs::path outDir;                                     // this run's own folder: <output dir>/<name>_<n>
        if (!MakeRunDir(baseDir, name, &outDir, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        SaveSettings(cfg, outDir / (name + ".config.ini"));

        AutoParams p = cfg.generate;
        p.gpu = nullptr;                                     // (the editor's GL context is the only tracer there is)
        p.startFrom = startFrom;
        auto live = std::make_shared<AutoLive>();
        live->ruleWeights = p.ruleWeights;
        if (!a.quiet)
            printf("Fitting %s: %d triangles, longest side %.1f mm.\n  %d steps, seed %u, %d views of %d px; results in %s\n", cfg.input.mesh.c_str(),
                   (int)mesh.indices.size() / 3, LongestSide(mesh), p.steps, p.seed, p.views, p.viewResolution, outDir.string().c_str());

        JobControl control;
        const auto start = Clock::now();
        int lastCheckpoint = -1;
        auto lastWritten = start;
        AutoResult r;
        {
            Watcher watch(control, cfg.runtime.progressSeconds,
                [&] {
                    std::lock_guard<std::mutex> l(live->mutex);
                    static bool said = false;
                    if (!said && live->iteration > 0 && !live->devices.empty() && !a.quiet) {
                        said = true;
                        printf("%s%s\n", live->devices.c_str(), live->deviceShortfall ? "  <- asked for the GPU, didn't get it" : "");
                    }
                    std::string stage = control.Stage();
                    if (live->iterations <= 0) return stage.empty() ? std::string("starting") : stage;
                    return Utils::Format("%s: step %d/%d (%.0f%%), loss %.4f, %d of %d rewrite rounds kept", stage.c_str(), live->iteration,
                                         live->iterations, 100.0f * live->iteration / live->iterations,
                                         live->history.empty() ? 0.0f : live->history.back(), live->accepted, live->rounds);
                },
                [&] {                                        // the best so far, now and then
                    if (cfg.output.checkpointSeconds <= 0) return;
                    auto now = Clock::now();
                    if (now - lastWritten < std::chrono::seconds(cfg.output.checkpointSeconds)) return;
                    Design best;
                    AutoScore score;
                    {
                        std::lock_guard<std::mutex> l(live->mutex);
                        if (live->version == 0 || live->version == lastCheckpoint) return;
                        lastCheckpoint = live->version;
                        best = live->best;
                        score = live->score;
                    }
                    lastWritten = now;
                    Write(outDir / (name + ".partial.txt"), SerializeWithTarget(best, score));      // (--start carries on from it)
                });
            r = AutoDesign(mesh, Design{}, p, control, live);
        }
        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        if (!r.ok) {
            fprintf(stderr, "Failed: %s\n", r.message.c_str());
            return r.cancelled ? 130 : 1;
        }

        // ---- the results
        std::vector<std::string> written;
        auto file = [&](const std::string& suffix) { return outDir / (name + suffix); };
        if (Write(file(".txt"), SerializeWithTarget(r.design, r.score))) written.push_back(file(".txt").string());
        {
            std::string t = Utils::Format("Mesh: %s (%d triangles)\nStart: %s\nSteps: %d   seed %u   time %s\n%s\n\n%s\n", cfg.input.mesh.c_str(),
                                          (int)mesh.indices.size() / 3, startFrom ? cfg.input.startDesign.c_str() : "the sphere, whole",
                                          p.steps, p.seed, Watcher::Duration(seconds).c_str(),
                                          r.cancelled ? "STOPPED EARLY: this is the best design found so far." : "Finished.",
                                          r.message.c_str());
            t += "\nScore\n" + ScoreLines(r.score);
            t += Utils::Format("\nMotion, checked at %d poses\n  overlap open %.2f mm^3, worst on the way %.2f mm^3 (up to %.2f mm deep)\n", r.checked.poses,
                               r.checked.openVolume, r.checked.worstVolume, r.checked.deepest);
            if (Write(file(".report.txt"), t)) written.push_back(file(".report.txt").string());
        }
        {
            std::ostringstream j;
            j << "{\n  \"mesh\": " << Json(cfg.input.mesh) << ",\n  \"start\": " << (startFrom ? Json(cfg.input.startDesign) : std::string("null"))
              << ",\n  \"ok\": true,\n  \"cancelled\": " << (r.cancelled ? "true" : "false")
              << ",\n  \"seconds\": " << Utils::Format("%.1f", seconds) << ",\n  \"steps\": " << p.steps << ",\n  \"seed\": " << p.seed
              << ",\n  \"pieces\": " << r.carving.pieces.size() << ",\n  \"joints\": " << r.carving.joints.size()
              << ",\n  \"score\": " << ScoreJson(r.score) << ",\n  \"motion_check\": "
              << Utils::Format("{\"poses\": %d, \"overlap_open_mm3\": %.6g, \"overlap_worst_mm3\": %.6g, \"deepest_mm\": %.6g}", r.checked.poses,
                               r.checked.openVolume, r.checked.worstVolume, r.checked.deepest)
              << ",\n  \"message\": " << Json(r.message) << "\n}\n";
            if (Write(file(".result.json"), j.str())) written.push_back(file(".result.json").string());
        }
        std::vector<float> history;
        std::vector<unsigned char> views;
        int vw = 0, vh = 0;
        {
            std::lock_guard<std::mutex> l(live->mutex);
            history = live->history;
            views = live->views;
            vw = live->viewsWidth;
            vh = live->viewsHeight;
        }
        if (cfg.output.history && !history.empty()) {
            std::string t = "evaluation,loss\n";
            for (size_t k = 0; k < history.size(); k++) t += Utils::Format("%d,%.6f\n", (int)k + 1, history[k]);
            if (Write(file(".history.csv"), t)) written.push_back(file(".history.csv").string());
        }
        if (cfg.output.viewsImage && vw > 0 && vh > 0 && views.size() == (size_t)vw * vh * 4) {
            std::vector<unsigned char> big;
            int bw, bh;
            Enlarge(views, vw, vh, &big, &bw, &bh);
            if (WritePng(file(".views.png").string(), big, bw, bh, 4, &err)) written.push_back(file(".views.png").string());
            else fprintf(stderr, "warning: %s\n", err.c_str());
        }
        if (cfg.output.preview) {
            if (WritePreviewPng(r.design, Placed(mesh, r.score.target), cfg.output.previewSize, file(".png").string(), &err)) written.push_back(file(".png").string());
            else fprintf(stderr, "warning: no preview picture: %s\n", err.c_str());
        }
        if (cfg.output.stl) {
            ExportParams ep = cfg.output.stlParams;
            const char* pose = ep.pose == ExportPose::Open ? "open" : (ep.pose == ExportPose::Closed ? "closed" : "design");
            ep.prefix = (outDir / (name + "_" + pose)).string();
            JobControl exportControl;
            ExportResult er;
            {
                Watcher watch(exportControl, cfg.runtime.progressSeconds, [&] { return "exporting STL: " + exportControl.Stage(); });
                er = ExportSTL(r.design, ep, exportControl);
            }
            if (er.ok) written.insert(written.end(), er.files.begin(), er.files.end());
            else fprintf(stderr, "warning: STL export: %s\n", er.message.c_str());
        }
        {
            std::error_code ec;
            fs::remove(file(".partial.txt"), ec);
        }

        printf("%s\n", r.message.c_str());
        printf("%s in %s. Written:\n", r.cancelled ? "Stopped" : "Done", Watcher::Duration(seconds).c_str());
        for (const std::string& f : written) printf("  %s\n", f.c_str());
        return r.cancelled ? 130 : 0;
    }

    // ---- score ------------------------------------------------------------------------------------------------------------
    int CmdScore(Cli::Config& cfg, const Args& a) {
        if (a.positional.size() != 2) { fprintf(stderr, "usage: foldaball score <design.txt> <mesh.obj>\n"); return 2; }
        std::string err;
        if (!cfg.Validate(&err)) { fprintf(stderr, "settings: %s\n", err.c_str()); return 2; }
        std::ifstream in(a.positional[0], std::ios::binary);
        if (!in) { fprintf(stderr, "can't open %s\n", a.positional[0].c_str()); return 1; }
        std::stringstream text;
        text << in.rdbuf();
        Design design;
        if (!design.Deserialize(text.str(), &err)) { fprintf(stderr, "%s: %s\n", a.positional[0].c_str(), err.c_str()); return 1; }
        TriMesh mesh;
        if (!LoadMesh(a.positional[1], cfg.input.meshSize, &mesh, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        AutoParams p = cfg.generate;
        p.gpu = nullptr;
        p.steps = 0;
        JobControl control;
        AutoResult r = AutoDesign(mesh, design, p, control, nullptr);
        if (!r.ok) { fprintf(stderr, "Failed: %s\n", r.message.c_str()); return 1; }
        printf("%s\n\nScore\n%s", r.message.c_str(), ScoreLines(r.score).c_str());
        const fs::path outDir = cfg.output.dir.empty() ? fs::path("foldaball_out") : fs::path(cfg.output.dir);
        if (MakeDir(outDir, &err)) {
            const std::string name = cfg.output.name.empty() ? fs::path(a.positional[0]).stem().string() : cfg.output.name;
            Write(outDir / (name + ".score.txt"), r.message + "\n\nScore\n" + ScoreLines(r.score));
        }
        return 0;
    }

    // ---- suite ------------------------------------------------------------------------------------------------------------
    int CmdSuite(Cli::Config& cfg, const Args& a) {
        if (a.positional.size() > 1) { fprintf(stderr, "suite takes one folder.\n"); return 2; }
        if (!a.positional.empty()) cfg.suite.folder = a.positional[0];
        std::string err;
        if (!cfg.Validate(&err)) { fprintf(stderr, "settings: %s\n", err.c_str()); return 2; }
        if (!cfg.input.startDesign.empty()) {
            fprintf(stderr, "start.design (--start) is for optimize: a suite starts every mesh from its sphere. Clear it (-s start.design=).\n");
            return 2;
        }
        const std::string folder = cfg.suite.folder;
        std::error_code ec;
        if (folder.empty() || !fs::is_directory(folder, ec)) { fprintf(stderr, "suite: '%s' is not a folder of .obj files.\n", folder.c_str()); return 2; }
        const size_t count = SuiteMeshes(folder).size();
        if (count == 0) { fprintf(stderr, "suite: no .obj files in %s.\n", folder.c_str()); return 2; }

        SuiteParams sp = cfg.suite;
        sp.folder = folder;
        sp.generate = cfg.generate;
        sp.generate.gpu = nullptr;
        sp.outDir = cfg.output.dir;
        sp.preview = cfg.output.preview;
        sp.previewSize = cfg.output.previewSize;
        const fs::path baseDir = sp.outDir.empty() ? fs::path(folder) / "autodesign_suite" : fs::path(sp.outDir);
        fs::path outDir;
        if (!MakeRunDir(baseDir, "suite", &outDir, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        sp.outDir = outDir.string(); 
        SaveSettings(cfg, outDir / "suite.config.ini");
        if (!a.quiet)
            printf("Test suite: %d mesh%s from %s, %d steps each%s; results in %s\n", (int)count, count == 1 ? "" : "es", folder.c_str(), sp.generate.steps,
                   sp.video ? ", with a video" : ", no video", outDir.string().c_str());

        auto live = std::make_shared<SuiteLive>();
        JobControl control;
        const auto start = Clock::now();
        SuiteResult r;
        {
            Watcher watch(control, cfg.runtime.progressSeconds, [&] {
                std::string phase, name;
                int index = 0, total = 0, it = 0, n = 0;
                float progress = 0.0f, loss = 0.0f;
                std::lock_guard<std::mutex> l(live->mutex);
                phase = live->phase;
                name = live->name;
                index = live->index;
                total = live->count;
                progress = live->progress;
                if (live->search) {
                    std::lock_guard<std::mutex> l2(live->search->mutex);
                    it = live->search->iteration;
                    n = live->search->iterations;
                    if (!live->search->history.empty()) loss = live->search->history.back();
                }
                if (phase == "Searching")
                    return Utils::Format("mesh %d/%d %s: step %d/%d, loss %.4f", index + 1, total, name.c_str(), it, n, loss);
                return Utils::Format("%s: %.0f%%", phase.empty() ? "starting" : phase.c_str(), progress * 100.0f);
            });
            r = RunSuite(sp, control, live);
        }
        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        if (r.ok) {
            printf("%s\n(%s)\n", r.message.c_str(), Watcher::Duration(seconds).c_str());
            return 0;
        }
        fprintf(stderr, "%s\n", r.message.c_str());
        return r.cancelled ? 130 : 1;
    }
}

int main(int argc, char** argv) {
    if (argc <= 1){
        Bakugan::App app;
        app.Run();
        return 0;
    }

    Args a;
    std::string err;
    if (!ParseArgs(argc, argv, &a, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 2; }
    if (a.help || a.command.empty() || a.command == "help") {
        fputs(kUsage, a.help || a.command == "help" ? stdout : stderr);
        return a.help || a.command == "help" ? 0 : 2;
    }
    if (a.command != "optimize" && a.command != "suite" && a.command != "score" && a.command != "config") {
        fprintf(stderr, "unknown command '%s' (see --help)\n", a.command.c_str());
        return 2;
    }
    SetTraceLogLevel(LOG_WARNING);                           // raylib's own chatter (image export): warnings only

    Cli::Config cfg;
    if (!Configure(a, cfg, &err)) { fprintf(stderr, "settings: %s\n", err.c_str()); return 2; }
    if (a.command == "config") {
        fputs(cfg.Dump().c_str(), stdout);
        return 0;
    }

    std::unique_ptr<tbb::global_control> threads;
    if (cfg.runtime.threads > 0)
        threads = std::make_unique<tbb::global_control>(tbb::global_control::max_allowed_parallelism, (size_t)cfg.runtime.threads);
    std::signal(SIGINT, OnSignal);
    std::signal(SIGTERM, OnSignal);

    if (a.command == "optimize") return CmdOptimize(cfg, a);
    if (a.command == "suite") return CmdSuite(cfg, a);
    return CmdScore(cfg, a);
}
