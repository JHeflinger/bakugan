#include "config.hpp"
#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace Bakugan::Cli {
    namespace {
        std::string Lower(std::string s) {
            for (char& c : s) c = c == '-' ? '_' : (char)tolower((unsigned char)c);
            return s;
        }

        std::string Trim(const std::string& s) {
            size_t a = 0, b = s.size();
            while (a < b && isspace((unsigned char)s[a])) a++;
            while (b > a && isspace((unsigned char)s[b - 1])) b--;
            return s.substr(a, b - a);
        }

        std::string Number(double v) {
            if (v == std::floor(v) && std::fabs(v) < 1e9) return std::to_string((long long)v);      // 100000, not 1e+05
            char buf[64];
            auto r = std::to_chars(buf, buf + sizeof(buf), v);      // the shortest text that reads back as the same number
            return std::string(buf, r.ptr);
        }

        std::string Number(float v) {
            if (v == std::floor(v) && std::fabs(v) < 1e9f) return std::to_string((long long)v);
            char buf[64];
            auto r = std::to_chars(buf, buf + sizeof(buf), v);
            return std::string(buf, r.ptr);
        }

        std::string Quote(const std::string& s) {
            bool plain = !s.empty() && s.find_first_of(" \t#;\"") == std::string::npos;
            return s.empty() || plain ? s : "\"" + s + "\"";
        }

        bool ParseInt(const std::string& v, long long* out) {
            if (v.empty()) return false;
            char* end = nullptr;
            errno = 0;
            long long x = strtoll(v.c_str(), &end, 10);
            if (errno != 0 || *end != '\0') return false;
            *out = x;
            return true;
        }

        bool ParseFloat(const std::string& v, double* out) {
            if (v.empty()) return false;
            char* end = nullptr;
            errno = 0;
            double x = strtod(v.c_str(), &end);
            if (errno != 0 || *end != '\0' || !std::isfinite(x)) return false;
            *out = x;
            return true;
        }

        int EditDistance(const std::string& a, const std::string& b) {
            std::vector<int> prev(b.size() + 1), cur(b.size() + 1);
            for (size_t j = 0; j <= b.size(); j++) prev[j] = (int)j;
            for (size_t i = 1; i <= a.size(); i++) {
                cur[0] = (int)i;
                for (size_t j = 1; j <= b.size(); j++)
                    cur[j] = std::min({ prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1) });
                std::swap(prev, cur);
            }
            return prev[b.size()];
        }

        std::string Slug(const std::string& name) {
            std::string s;
            for (char c : name) s += isalnum((unsigned char)c) ? (char)tolower((unsigned char)c) : '_';
            return s;
        }
    }

    Config::Config() {
        // the grammar's weights are a vector the optimizer reads; sized as the editor sizes it, at the defaults
        for (const AutoRuleInfo& r : AutoRules()) generate.ruleWeights.push_back(r.weight);
        suite.meshSize = 80.0f;          // (as the suite's own default)
        Build();
    }

    // ---- the settings -----------------------------------------------------------------------------------------------------

    void Config::Build() {
        std::string section;
        auto add = [&](const char* name, std::string help, std::string range, std::function<bool(const std::string&, std::string*)> set,
                       std::function<std::string()> get) {
            m_Options.push_back({ section, name, std::move(help), std::move(range), std::move(set), std::move(get) });
        };
        auto Bool = [&](const char* name, bool& ref, const char* help) {
            add(name, help, "true | false",
                [&ref](const std::string& v, std::string* err) {
                    std::string s = Lower(v);
                    if (s == "true" || s == "yes" || s == "on" || s == "1") ref = true;
                    else if (s == "false" || s == "no" || s == "off" || s == "0") ref = false;
                    else { *err = "expected true or false, got '" + v + "'"; return false; }
                    return true;
                },
                [&ref] { return std::string(ref ? "true" : "false"); });
        };
        auto Str = [&](const char* name, std::string& ref, const char* help) {
            add(name, help, "text",
                [&ref](const std::string& v, std::string*) { ref = v; return true; },
                [&ref] { return Quote(ref); });
        };
        auto Int = [&](const char* name, auto& ref, long long lo, long long hi, const char* help) {
            using T = std::remove_reference_t<decltype(ref)>;
            add(name, help, std::to_string(lo) + ".." + std::to_string(hi),
                [&ref, lo, hi](const std::string& v, std::string* err) {
                    long long x;
                    if (!ParseInt(v, &x)) { *err = "expected a whole number, got '" + v + "'"; return false; }
                    if (x < lo || x > hi) { *err = "must be between " + std::to_string(lo) + " and " + std::to_string(hi) + ", got " + v; return false; }
                    ref = (T)x;
                    return true;
                },
                [&ref] { return std::to_string((long long)ref); });
        };
        // zero: the word that means 0 for a number where 0 is a setting of its own ("auto", "off"), shown in the range
        auto Float = [&](const char* name, float& ref, double lo, double hi, const char* help, const char* zero = nullptr) {
            std::string range = Number(lo) + ".." + Number(hi);
            if (zero) range += std::string("; 0 = ") + zero;
            std::string word = zero ? Lower(std::string(zero).substr(0, std::string(zero).find(':'))) : "";     // "auto: the mesh's volume" -> auto
            add(name, help, range,
                [&ref, lo, hi, word](const std::string& v, std::string* err) {
                    double x;
                    std::string s = Lower(v);
                    if (lo <= 0.0 && (s == "off" || s == "none" || (!word.empty() && s == word))) x = 0.0;
                    else if (!ParseFloat(v, &x)) { *err = "expected a number, got '" + v + "'"; return false; }
                    if (x < lo || x > hi) { *err = "must be between " + Number(lo) + " and " + Number(hi) + ", got " + v; return false; }
                    ref = (float)x;
                    return true;
                },
                [&ref] { return Number(ref); });
        };
        auto Choice = [&](const char* name, auto& ref, std::vector<std::pair<std::string, int>> choices, const char* help) {
            using E = std::remove_reference_t<decltype(ref)>;
            std::string range;
            for (auto& c : choices) range += (range.empty() ? "" : " | ") + c.first;
            add(name, help, range,
                [&ref, choices, range](const std::string& v, std::string* err) {
                    std::string s = Lower(v);
                    for (auto& c : choices)
                        if (s == c.first) { ref = (E)c.second; return true; }
                    *err = "expected " + range + ", got '" + v + "'";
                    return false;
                },
                [&ref, choices] {
                    for (auto& c : choices)
                        if ((int)ref == c.second) return c.first;
                    return std::string();
                });
        };

        AutoParams& p = generate;

        // -- the run ------------------------------------------------------------------------------------------------------
        section = "input";
        Str("mesh", input.mesh, "The .obj to fit (optimize, score). Can also be given on the command line.");
        Float("mesh_size", input.meshSize, 0, 100000,
              "Centre the mesh and scale its longest side to this many mm before fitting. 0 leaves it as the file has it.", "as is");

        section = "output";
        Str("dir", output.dir, "Where the results go. Empty: ./foldaball_out (optimize, score) or <suite folder>/autodesign_suite.");
        Str("name", output.name, "File name stem of the results. Empty: the mesh's own.");
        Bool("preview", output.preview, "Write <name>.png: the result open, a third and two thirds closed, and closed (software rendered).");
        Int("preview_size", output.previewSize, 64, 2048, "Pixels across each pose in that picture.");
        Bool("views_image", output.viewsImage, "Write <name>.views.png: the silhouettes scored last, against the placed mesh's (as in the Generate tab).");
        Bool("history", output.history, "Write <name>.history.csv: the loss at every step of the run.");
        Int("checkpoint_seconds", output.checkpointSeconds, 0, 86400,
            "Save the best design so far to <name>.partial.txt this often, so a killed run loses little. 0: never.");
        Bool("stl", output.stl, "Also export the result as STL parts (<name>_<pose>_<part>.stl). Slow for fine voxels.");
        Choice("stl_pose", output.stlParams.pose,
               { { "open", (int)ExportPose::Open }, { "closed", (int)ExportPose::Closed }, { "design", (int)ExportPose::Design } },
               "The pose the STL parts are exported in.");
        Float("stl_voxel", output.stlParams.voxel, 0.02, 5.0, "STL mesh voxel size, mm: smaller is finer and much slower.");
        Float("stl_adaptivity", output.stlParams.adaptivity, 0, 1, "OpenVDB mesh simplification for the STL parts.");
        Bool("stl_separate", output.stlParams.separate, "One STL per part.");
        Bool("stl_combined", output.stlParams.combined, "One STL with every part together (print in place).");
        Bool("stl_keep_apart", output.stlParams.keepApart, "Trim bodies that touch in the export pose, so parts printed together come out apart.");
        Float("stl_gap", output.stlParams.gap, 0, 5, "... to this gap, mm.");

        section = "runtime";
        Int("threads", runtime.threads, 0, 4096, "Worker threads for the search (0: every core).");
        Int("progress_seconds", runtime.progressSeconds, 0, 86400, "Print a progress line this often (0: none).");

        // -- the Generate tab -----------------------------------------------------------------------------------------------
        section = "goal";
        Int("views", p.views, 1, 64, "Directions the outlines are compared from, spread over a half sphere. (The slider: 1..16)");
        Int("view_pixels", p.viewResolution, 8, 1024, "Pixels across each view: finer is more exact and slower. (The slider: 32..192)");
        Float("reach_weight", p.reachWeight, 0, 1000, "What parts of the mesh's outline being far from the design's cost: a limb nothing is near pulls the nearest piece toward it.", "ignored");
        Float("min_view_coverage", p.minViewCoverage, 0, 1,
              "Share of every view's target outline the design must cover; each view short of it costs steeply.");
        Float("collision_weight", p.collisionWeight, 0, 1000, "What pieces overlapping each other costs, in the open pose and in every hinge's turn.", "ignored");
        Float("collision_ramp", p.collisionRamp, 1, 10000,
              "Overlaps cost this many times as much by the end of the search as at its start, rising steadily. 1: the same throughout.");
        Int("hinge_checks", p.hingeChecks, 1, 256, "Poses checked for overlaps in every hinge's turn, spread evenly over its angle.");
        Bool("order_hinges", p.orderHinges, "Hinges open one after another, the freest first. Off: each hinge's timing is tuned by Adam.");
        Bool("dependencies", p.dependencies, "Every hinge but the first to open is held shut by a piece that moves away first (implies order_hinges).");
        Float("dependency_weight", p.dependencyWeight, 0, 1000, "What a hinge with nothing near enough to hold it costs.");
        Float("ledge_size", p.ledges.size, 0, 1000, "mm a ledge reaches over the piece it holds (dependencies).", "auto: 12% of the sphere's radius");
        Float("ledge_reach", p.ledges.reach, 0, 1000, "mm of another piece's surface a ledge may cross to get there (dependencies).", "auto: 35% of the radius");
        Float("hinge_weight", p.hingeWeight, 0, 1000, "What a hinge being away from the two pieces it joins costs. It is also what limits the search most.", "ignored");
        Float("thin_weight", p.thinWeight, 0, 1000, "What a piece thinner than min_thickness costs (1 - its thickness / the minimum).", "ignored");
        Float("min_thickness", p.minThickness, 0.5, 1000, "mm: the thinnest a piece may be (the largest ball that fits in it, across).");
        Bool("free_scale", p.freeScale, "The mesh's scale is optimized too: the search may shrink the mesh it's compared with (easier to match, bulkier result).");
        Bool("move_target", p.moveTarget, "The mesh the design is compared with may be moved to wherever suits it best.");
        Bool("scale_target", p.scaleTarget, "... and scaled (0.3x .. 3.3x) about its centre of volume.");

        section = "start";
        Float("core", p.core, 0, 0.95, "Start from a shell and a core inside it, this share of the sphere across: the shell makes thin parts, the core chunky ones.", "whole sphere");
        Float("shell_first", p.shellFirst, 0, 1, "With a core: this share of the steps only the shell is cut (the outline); then the core is free and thickens the shell's pieces.", "both at once");
        Float("sphere_diameter", p.sphereDiameter, 0, 100000, "The sphere that's carved, mm across, at the mesh's centre of volume.", "auto: the mesh's volume");

        section = "semantic";
        Float("semantic_weight", p.semanticWeight, 0, 1000, "What the views looking different to DINOv3 costs. Needs the model and ONNX Runtime below.", "off");
        Float("silhouette_weight", p.silhouetteWeight, 0, 1000, "What the outline comparison costs, beside it. 0: only DINOv3 judges.", "off");
        Str("semantic_model", p.semanticModel, "The .onnx that tools/autodesign/export_dinov3.py writes (it sets the views' size).");
        Str("onnx_runtime", p.onnxRuntime, "The ONNX Runtime library (libonnxruntime.so ...). Empty: found as usual.");
        Bool("semantic_gpu", p.semanticGpu, "Run DINOv3 with CUDA, if this ONNX Runtime has it. Off on a server without a GPU.");
        Choice("semantic_look", p.semanticLook,
               { { "silhouette", (int)SemanticLook::Silhouette }, { "depth", (int)SemanticLook::Depth }, { "shaded", (int)SemanticLook::Shaded } },
               "How the design and the mesh are drawn for DINOv3.");
        Float("patch_weight", p.patchWeight, 0, 1, "0: compare the global (CLS) embeddings; 1: the patch embeddings, place by place; between: a mix.");
        Int("semantic_samples", p.semanticSamples, 0, 1000000, "Covered pixels whose shading the gradient follows each step.");

        section = "optimizer";
        Int("steps", p.steps, 0, 100000000, "Adam steps in all (each one gradient), the rewrites' lookahead included: the budget. (The slider: 100..20000)");
        Float("learning_rate", p.learningRate, 0.0001, 100, "Adam's step size, in each parameter's own unit. (The slider: 0.05..5)");
        Int("steps_per_round", p.stepsPerRound, 1, 1000000, "Adam steps between rewrite rounds.");
        Int("rewrites_per_round", p.rewritesPerRound, 0, 1000, "Grammar rewrites tried in a round (drawn by the [grammar] weights).");
        Int("lookahead", p.lookahead, 1, 100000, "Adam steps every rewrite, and the design without one, get before they are compared.");
        Float("rewrite_margin", p.rewriteMargin, 0, 1,
              "How much better (a share of the loss) a rewrite that adds something must do to be kept; how much worse one that removes may.");
        Int("max_pieces", p.maxPieces, 2, 4096, "Adding a cut makes one more piece: no more than this.");
        Int("seed", p.seed, 0, 4294967295LL, "Same seed and settings, same result.");

        section = "grammar";
        {
            // one weight per rewrite rule, named as in the editor's Grammar section
            std::vector<AutoRuleInfo> rules = AutoRules();
            for (size_t i = 0; i < rules.size(); i++) {
                float& ref = p.ruleWeights[i];
                std::string key = Slug(rules[i].name);
                Option o;
                o.section = section;
                o.name = key;
                o.help = std::string("How often \"") + rules[i].name + "\" is drawn for a rewrite (0: never). " +
                         (rules[i].simplifies ? "Takes something away: kept when no worse." : "Adds something: kept when better.");
                o.range = "0..1000";
                o.set = [&ref](const std::string& v, std::string* err) {
                    double x;
                    if (!ParseFloat(v, &x)) { *err = "expected a number, got '" + v + "'"; return false; }
                    if (x < 0 || x > 1000) { *err = "must be between 0 and 1000, got " + v; return false; }
                    ref = (float)x;
                    return true;
                };
                o.get = [&ref] { return Number(ref); };
                m_Options.push_back(std::move(o));
            }
        }

        section = "finishing";
        Bool("build_hinges", p.buildHinges, "Give every hinge the smallest barrel that prints, when the search is done; the search then keeps each axis where one fits.");
        Bool("hinge_flush", p.hinges.flush, "Each axis far enough under the sphere's surface for its barrel to be inside the sphere.");
        Float("hinge_scale", p.hinges.scale, 0.1, 10, "Hinge size against the smallest that prints (a 1.6 mm pin in a 3.8 mm barrel). (The slider: 0.75..2.5)");
        Float("hinge_reach", p.hinges.reach, 0, 100, "mm a piece may be from a hinge's axis and still be joined to it by a leaf that long.", "smallest");
        Float("hinge_clearance", p.hinges.clearance, 0, 10, "mm between the pin and the knuckles around it, and between knuckles.");
        Choice("hinge_pin", p.hinges.pin, { { "fused", (int)PinStyle::Fused }, { "separate", (int)PinStyle::Separate } },
               "fused: printed in place. separate: both bored for a pin pushed in afterwards.");
        Bool("carve_when_done", p.carveWhenDone, "Carve away whatever still overlaps on the way between open and closed.");
        Float("carve_clearance", p.carve.clearance, 0, 10, "mm of room left around what is carved away.");
        Int("carve_checks", p.carve.checksPerHinge, 1, 1000, "Poses looked at in every hinge's turn when carving.");
        Int("carve_rounds", p.carve.rounds, 1, 100, "Carve, check again between those poses, carve what is left, ... up to this many times.");
        Bool("latch_when_done", p.latchWhenDone, "A press latch on the hinge that opens first.");
        Float("latch_scale", p.latch.scale, 0.1, 10, "Latch size against the default for the sphere. (The slider: 0.6..1.6)");
        Float("latch_clearance", p.latch.clearance, 0, 10, "mm printed around the rocker.");

        section = "suite";
        Str("folder", suite.folder, "The folder of .obj files (suite). Can also be given on the command line.");
        Float("mesh_size", suite.meshSize, 0, 100000, "Each mesh is centred and scaled so its longest side is this long, mm.", "as is");
        Bool("video", suite.video, "Render the video (needs ffmpeg). Off: designs, pictures and summary.csv only.");
        Int("video_size", suite.videoSize, 256, 8192, "Pixels, square.");
        Int("fps", suite.fps, 1, 120, "Frames per second.");
        Float("search_seconds", suite.searchSeconds, 0.1, 600, "Video time for the search (one frame per evenly spaced snapshot).");
        Float("motion_seconds", suite.motionSeconds, 0.1, 600, "Video time for the closing, and again for the opening.");
        Float("hold_seconds", suite.holdSeconds, 0, 600, "Pauses on the result, closed, and open again.");
    }

    // ---- reading and writing --------------------------------------------------------------------------------------------

    const Option* Config::Find(const std::string& key) const {
        std::string k = Lower(key);
        for (const Option& o : m_Options)
            if (o.Key() == k) return &o;
        return nullptr;
    }

    std::string Config::Suggest(const std::string& key) const {
        std::string k = Lower(key), best;
        int bestDistance = 1 << 30;
        for (const Option& o : m_Options) {
            int d = EditDistance(k, o.Key());
            // a bare name without its section ("steps") is a good hint too
            d = std::min(d, EditDistance(k, o.name) + 1);
            if (d < bestDistance) { bestDistance = d; best = o.Key(); }
        }
        return bestDistance <= std::max<int>(3, (int)k.size() / 3) ? best : std::string();
    }

    bool Config::Set(const std::string& key, const std::string& value, std::string* error) {
        const Option* o = Find(key);
        if (!o) {
            *error = "unknown setting '" + key + "'";
            std::string hint = Suggest(key);
            if (!hint.empty()) *error += " (did you mean '" + hint + "'?)";
            return false;
        }
        std::string why;
        if (!o->set(Trim(value), &why)) {
            *error = o->Key() + ": " + why;
            return false;
        }
        return true;
    }

    bool Config::LoadFile(const std::string& path, std::string* error) {
        std::ifstream in(path);
        if (!in) { *error = "can't open " + path; return false; }
        std::string line, section;
        int number = 0;
        while (std::getline(in, line)) {
            number++;
            auto fail = [&](const std::string& why) {
                *error = path + ":" + std::to_string(number) + ": " + why;
                return false;
            };
            std::string s = Trim(line);
            if (s.empty() || s[0] == '#' || s[0] == ';') continue;
            if (s[0] == '[') {
                size_t close = s.find(']');
                if (close == std::string::npos) return fail("missing ']'");
                section = Lower(Trim(s.substr(1, close - 1)));
                continue;
            }
            size_t eq = s.find('=');
            if (eq == std::string::npos) return fail("expected 'key = value', got '" + s + "'");
            std::string key = Lower(Trim(s.substr(0, eq)));
            std::string value = Trim(s.substr(eq + 1));
            if (!value.empty() && value[0] == '"') {                 // "quoted": everything up to the closing quote
                size_t end = value.find('"', 1);
                if (end == std::string::npos) return fail("missing closing quote");
                value = value.substr(1, end - 1);
            } else {                                                  // a comment after the value: " # ..." or " ; ..."
                for (size_t k = 1; k < value.size(); k++)
                    if ((value[k] == '#' || value[k] == ';') && isspace((unsigned char)value[k - 1])) { value = Trim(value.substr(0, k)); break; }
            }
            if (key.find('.') == std::string::npos) {
                if (section.empty()) return fail("'" + key + "' is outside any [section]: write it as section." + key);
                key = section + "." + key;
            }
            std::string why;
            if (!Set(key, value, &why)) return fail(why);
        }
        return true;
    }

    bool Config::Validate(std::string* error) const {
        if (generate.semanticWeight > 0.0f && generate.semanticModel.empty()) {
            *error = "semantic.semantic_weight is on but semantic.semantic_model is not set (the .onnx tools/autodesign/export_dinov3.py writes)";
            return false;
        }
        if (generate.semanticWeight <= 0.0f && generate.silhouetteWeight <= 0.0f) {
            *error = "semantic.silhouette_weight and semantic.semantic_weight are both off: there is nothing to optimize";
            return false;
        }
        if (suite.fps < 1) { *error = "suite.fps must be at least 1"; return false; }
        return true;
    }

    std::string Config::Dump() const {
        std::ostringstream out;
        out << "# foldaball settings. Written by `foldaball config`: every setting, at its current value.\n"
               "# Lines starting with # or ; are comments. Delete any line to keep the editor's default for it.\n"
               "# Paths are relative to the directory foldaball is run from.\n";
        std::string section;
        for (const Option& o : m_Options) {
            if (o.section != section) {
                section = o.section;
                out << "\n[" << section << "]\n";
            }
            out << "\n# " << o.help << "\n# (" << o.range << ")\n" << o.name << " = " << o.get() << "\n";
        }
        return out.str();
    }
}
