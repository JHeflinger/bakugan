// The Generate tab's jobs: the generator (autodesign/) and the test suite running on worker threads, and what the
// editor does with their progress and results.
#include "app.hpp"
#include "util/utils.hpp"
#include <nfd.h>
#include <cstdio>
#include <cstdlib>

namespace Bakugan {
    // ---- the generator ------------------------------------------------------------------------------------------------

    void App::StartAuto(bool scoreOnly) {
        if (m_AutoJob.Busy()) return;
        if (m_SuiteJob.Busy()) { m_Status = "A test suite is running: wait for it or stop it first."; return; }
        if (!m_HasReference) {
            m_Status = "Load the mesh to fit as the reference mesh first (Model tab > Reference mesh).";
            return;
        }
        if (scoreOnly && m_Design.shapes.empty()) { m_Status = "Nothing to score: the design is empty."; return; }
        // the mesh as it is placed in the design view (scale, offset, rotation)
        auto mesh = std::make_shared<const TriMesh>(Utils::ModelToTriMesh(m_Reference, ReferenceMatrix()));
        if (mesh->indices.empty()) { m_Status = "The reference mesh has no triangles to fit."; return; }
        AutoParams p = m_AutoParams;
        if (scoreOnly) {
            p.steps = 0;                 // only score the start: the current design
            p.fromCurrent = true;
        }
        if (p.fromCurrent && m_Design.shapes.empty()) p.fromCurrent = false;
        Design start = m_Design;
        m_AutoLive = std::make_shared<AutoLive>();
        m_AutoLive->ruleWeights = p.ruleWeights;      // the Grammar sliders change these while it runs
        std::shared_ptr<AutoLive> live = m_AutoLive;
        m_AutoSeen = -1;
        m_AutoScoreOnly = scoreOnly;
        m_AutoHistory.clear();
        m_AutoStart = GetTime();
        m_Anim = 0;
        // the parts and the collision check would be stale many times over: they resume when this is done
        if (!scoreOnly) {
            m_BuildJob.Cancel();
            m_SweepJob.Cancel();
        }
        m_AutoJob.Start([mesh, start, p, live](JobControl& c) { return AutoDesign(*mesh, start, p, c, live); });
        m_Status = scoreOnly ? "Scoring the design against " + m_ReferenceName + "..."
                             : "Generating a design for " + m_ReferenceName + (p.fromCurrent ? " from the current one..." : "...");
    }

    // The scored silhouettes of the best design, as a texture for the Generate tab (on this thread: it has the GL
    // context). A new size makes a new texture.
    static void UploadViews(const std::vector<unsigned char>& rgba, int w, int h, Texture2D* tex, bool* has) {
        if (rgba.empty() && *has) {                     // none this time (e.g. the semantic loss is off now)
            UnloadTexture(*tex);
            *has = false;
        }
        if (w <= 0 || h <= 0 || rgba.size() != (size_t)w * h * 4) return;
        if (*has && (tex->width != w || tex->height != h)) {
            UnloadTexture(*tex);
            *has = false;
        }
        if (!*has) {
            Image img = { (void*)rgba.data(), w, h, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 };
            *tex = LoadTextureFromImage(img);
            SetTextureFilter(*tex, TEXTURE_FILTER_POINT);
            *has = IsTextureValid(*tex);
        } else {
            UpdateTexture(*tex, rgba.data());
        }
    }

    static void UploadViews(const AutoLive& live, Texture2D* views, bool* hasViews, Texture2D* semantic, bool* hasSemantic) {
        UploadViews(live.views, live.viewsWidth, live.viewsHeight, views, hasViews);
        UploadViews(live.semanticViews, live.semanticWidth, live.semanticHeight, semantic, hasSemantic);
    }

    void App::PollAuto() {
        // while it runs: the best design so far, its score and views
        if (m_AutoLive && m_AutoJob.Busy()) {
            std::lock_guard<std::mutex> l(m_AutoLive->mutex);
            m_AutoHistory = m_AutoLive->history;
            if (m_AutoLive->version != m_AutoSeen) {
                m_AutoSeen = m_AutoLive->version;
                m_AutoScore = m_AutoLive->score;
                m_HasAutoScore = m_AutoScore.valid;
                UploadViews(*m_AutoLive, &m_AutoViews, &m_HasAutoViews, &m_AutoSemantic, &m_HasAutoSemantic);
                if (!m_AutoScoreOnly && m_AutoPreview) {
                    m_Design = m_AutoLive->best;
                    m_AutoScoreFor = m_Design;
                    SanitizeSelection();
                }
            }
        }
        // when it is done: the result
        AutoResult r;
        if (!m_AutoJob.Poll(&r)) return;
        if (!m_AutoScoreOnly && !r.cancelled) Notify();
        if (m_AutoLive) {
            std::lock_guard<std::mutex> l(m_AutoLive->mutex);
            m_AutoHistory = m_AutoLive->history;
            UploadViews(*m_AutoLive, &m_AutoViews, &m_HasAutoViews, &m_AutoSemantic, &m_HasAutoSemantic);
        }
        m_Status = r.message;
        if (!r.ok) return;
        m_AutoScore = r.score;
        m_HasAutoScore = r.score.valid;
        if (m_AutoScoreOnly) {
            m_AutoScoreFor = m_Design;
            return;
        }
        m_Design = r.design;
        m_AutoScoreFor = m_Design;
        m_Sel = Selection{};
        m_T = 0.0f;
        m_Anim = 0;
        m_Motion = Motion::Closing;
        m_HasSweep = false;
        FrameView(m_OutputView);
        m_Status += " Space plays the fold; Ctrl+Z brings back the design from before.";
    }

    void App::ClearAutoScore() {
        m_HasAutoScore = false;
        m_AutoHistory.clear();
        if (m_HasAutoViews) UnloadTexture(m_AutoViews);
        m_HasAutoViews = false;
        if (m_HasAutoSemantic) UnloadTexture(m_AutoSemantic);
        m_HasAutoSemantic = false;
    }

    // ---- the test suite -------------------------------------------------------------------------------------------------

    void App::BrowseSuiteFolder() {
        std::string start = m_SuiteFolder[0] ? std::string(m_SuiteFolder) : std::string(GetWorkingDirectory());
        if (!DirectoryExists(start.c_str())) start = GetWorkingDirectory();
        nfdchar_t* path = nullptr;
        nfdresult_t r = NFD_PickFolder(start.c_str(), &path);
        SetWindowFocused();
        if (r == NFD_OKAY && path) {
            snprintf(m_SuiteFolder, sizeof(m_SuiteFolder), "%s", path);
            free(path);
        } else if (r == NFD_ERROR) {
            const char* e = NFD_GetError();
            m_Status = std::string("The folder dialog failed") + (e ? std::string(": ") + e : "") + ". You can type the path instead.";
        }
    }

    void App::StartSuite() {
        if (m_SuiteJob.Busy() || m_AutoJob.Busy()) return;
        std::string folder = m_SuiteFolder;
        if (folder.empty() || !DirectoryExists(folder.c_str())) { m_Status = "Choose a folder of .obj files for the test suite."; return; }
        size_t count = SuiteMeshes(folder).size();
        if (count == 0) { m_Status = "There are no .obj files in " + folder + "."; return; }
        SuiteParams p = m_SuiteParams;
        p.folder = folder;
        p.generate = m_AutoParams;
        p.generate.fromCurrent = false;
        m_SuiteLive = std::make_shared<SuiteLive>();
        std::shared_ptr<SuiteLive> live = m_SuiteLive;
        m_SuiteStart = GetTime();
        m_SuiteJob.Start([p, live](JobControl& c) { return RunSuite(p, c, live); });
        m_Status = Utils::Format("Test suite: %d meshes from %s...", (int)count, folder.c_str());
    }

    void App::PollSuite() {
        SuiteResult r;
        if (!m_SuiteJob.Poll(&r)) return;
        m_Status = r.message;
        if (r.ok) m_SuiteVideo = r.video;
        if (!r.cancelled) Notify();
    }

    // notification.mp3 from the working directory or next to the executable, looked for again each time until found
    // (so it can be dropped in while the editor runs). Silent without it or without an audio device.
    void App::Notify() {
        if (!m_Audio) return;
        if (!m_HasNotifySound) {
            const std::string candidates[] = { "notification.mp3", std::string(GetApplicationDirectory()) + "notification.mp3" };
            for (const std::string& path : candidates) {
                if (!FileExists(path.c_str())) continue;
                m_NotifySound = LoadSound(path.c_str());
                m_HasNotifySound = IsSoundValid(m_NotifySound);
                if (m_HasNotifySound) break;
            }
            if (!m_HasNotifySound) return;
        }
        PlaySound(m_NotifySound);
    }
}
