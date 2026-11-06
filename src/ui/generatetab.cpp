// The Generate tab: the input mesh, the generator's settings, its progress and score, and the test suite.
#include "core/app.hpp"
#include "ui/widgets.hpp"
#include <imgui.h>
#include <rlImGui.h>
#include <raymath.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>

namespace Bakugan {
    using namespace Widgets;

    void App::DrawGenerateTab() {
        Hint("Builds a design from the reference mesh: superfrustum bodies on hinges whose silhouettes match the mesh's when "
             "open and a sphere's when closed. Adam tunes every shape and hinge value; a small grammar adds bodies.");
        GenerateInput();
        ImGui::BeginDisabled(m_AutoJob.Busy());
        GenerateGoal();
        GenerateStart();
        GenerateOptimizer();
        ImGui::EndDisabled();
        GenerateRun();
        if (m_HasAutoScore) GenerateScore();
        GenerateSuite();
    }

    void App::GenerateInput() {
        ImGui::SeparatorText("Input mesh");
        bool busy = m_AutoJob.Busy() || m_SuiteJob.Busy();
        if (!m_HasReference) {
            Warn("Load the mesh to turn into a Bakugan as the reference mesh. It is fitted where it is placed in the design view.");
            ImGui::BeginDisabled(busy);
            if (ImGui::Button("Import mesh...", ImVec2(-FLT_MIN, 0))) BrowseReference();
            ImGui::EndDisabled();
            return;
        }
        BoundingBox b = ReferenceBounds();
        Vector3 e = Vector3Subtract(b.max, b.min);
        ImGui::TextUnformatted(m_ReferenceName.c_str());
        ImGui::SameLine();
        ImGui::TextDisabled("%.0f x %.0f x %.0f mm", e.x, e.y, e.z);
        ImGui::SetItemTooltip("Scale, move and turn it in the Model tab's Reference mesh panel; the design is fitted to it there");
        // another mesh can be loaded any time a search isn't using this one
        ImGui::BeginDisabled(busy);
        if (ImGui::Button("Import another mesh...", ImVec2(HalfWidth(), 0))) BrowseReference();
        ImGui::SetItemTooltip("Replace the input mesh (the design stays; generate again to fit the new mesh)");
        ImGui::SameLine();
        if (ImGui::Button("Remove mesh", ImVec2(-FLT_MIN, 0))) RemoveReference();
        ImGui::EndDisabled();
    }

    void App::GenerateGoal() {
        AutoParams& p = m_AutoParams;
        ImGui::SeparatorText("Goal");
        float w = ImGui::GetContentRegionAvail().x;
        ImGui::TextDisabled("match the mesh");
        ImGui::SameLine(w - ImGui::CalcTextSize("close into a sphere").x);
        ImGui::TextDisabled("close into a sphere");
        char fmt[64];
        int sphere = (int)lroundf(p.sphereBias * 100.0f);
        snprintf(fmt, sizeof(fmt), "fit %d%%%%  |  sphere %d%%%%", 100 - sphere, sphere);
        ImGui::SetNextItemWidth(-FLT_MIN);
        ImGui::SliderFloat("##bias", &p.sphereBias, 0.0f, 1.0f, fmt);
        ImGui::SetItemTooltip("How much each half of the score counts: how well the open design matches the mesh,\n"
                              "against how round the closed design is");
        {
            ImGui::SliderInt("Views", &p.views, 1, 16);
            ImGui::SetItemTooltip("Directions the outlines are compared from, spread over a half sphere (from the other\n"
                                  "half they would only be mirror images)");
            ImGui::SliderInt("View pixels", &p.viewResolution, 32, 192);
            ImGui::SetItemTooltip("Pixels across each view: finer is more exact and slower");
            ImGui::SliderFloat("Min coverage", &p.minViewCoverage, 0.0f, 1.0f, "%.2f");
            ImGui::SetItemTooltip("Share of every view's target outline the design must cover. Each view short of it costs\n"
                                  "steeply (10 per whole share missing), so no single view can be given up.");
        }
        ImGui::DragFloat("Sphere size", &p.sphereDiameter, 0.5f, 0.0f, 500.0f, p.sphereDiameter > 0.0f ? "%.1f mm across" : "auto");
        ImGui::SetItemTooltip("Diameter the closed design should have. Auto: a sphere of the mesh's volume, at its centre.");
    }

    void App::GenerateStart() {
        AutoParams& p = m_AutoParams;
        ImGui::SeparatorText("Start from");
        if (ImGui::RadioButton("Two hinged boxes", !p.fromCurrent)) p.fromCurrent = false;
        ImGui::SetItemTooltip("The mesh's box halved along its longest side: a box in each half, a body each, hinged to fold");
        ImGui::SameLine();
        ImGui::BeginDisabled(m_Design.shapes.empty());
        if (ImGui::RadioButton("The current design", p.fromCurrent)) p.fromCurrent = true;
        ImGui::SetItemTooltip("Refine what is in the editor");
        ImGui::EndDisabled();
    }

    void App::GenerateOptimizer() {
        AutoParams& p = m_AutoParams;
        ImGui::SeparatorText("Optimizer");
        ImGui::SliderInt("Steps", &p.steps, 100, 20000, "%d", ImGuiSliderFlags_Logarithmic);
        ImGui::SetItemTooltip("Adam steps in all (each one gradient), the rewrites' lookahead steps included: the budget");
        ImGui::SliderFloat("Learning rate", &p.learningRate, 0.05f, 5.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
        ImGui::SetItemTooltip("Adam's step size, in each parameter's own unit (0.5%% of the mesh's size for lengths,\n"
                              "0.01 for rotations, angles and profile values)");
        ImGui::SliderInt("Steps per round", &p.stepsPerRound, 5, 500);
        ImGui::SetItemTooltip("Adam steps between rewrite rounds");
        ImGui::SliderInt("Rewrites per round", &p.rewritesPerRound, 0, 12);
        ImGui::SetItemTooltip("Grammar rewrites tried in a round (now: adding a small superfrustum on a hinge)");
        ImGui::SliderInt("Lookahead", &p.lookahead, 1, 100);
        ImGui::SetItemTooltip("Adam steps every rewrite, and the design without one, get before they are compared:\n"
                              "the best of them carries on");
        ImGui::SliderInt("Max bodies", &p.maxBodies, 2, 16);
        int seed = (int)p.seed;
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.5f);
        if (ImGui::InputInt("Seed", &seed)) p.seed = (unsigned)std::max(seed, 0);
        ImGui::SetItemTooltip("Same seed and settings, same result");
        ImGui::SameLine();
        if (ImGui::Button("Shuffle")) p.seed = (unsigned)GetRandomValue(1, 999999);
    }

    // The buttons, or the progress of a run.
    void App::GenerateRun() {
        ImGui::Checkbox("Show progress live", &m_AutoPreview);
        ImGui::SetItemTooltip("Put the best design so far in the editor while the search runs");
        ImGui::SameLine();
        ImGui::Checkbox("Show sphere", &m_ShowAutoSphere);
        ImGui::SetItemTooltip("The sphere the closed design is scored against, in the output view");
        if (!m_AutoJob.Busy()) {
            ImGui::BeginDisabled(!m_HasReference || m_SuiteJob.Busy());
            if (ImGui::Button("Generate", ImVec2(HalfWidth(), 30))) StartAuto(false);
            ImGui::SetItemTooltip(m_AutoParams.fromCurrent ? "Refine the current design (Ctrl+Z brings it back)"
                                                           : "Replace the design with a generated one (Ctrl+Z brings it back)");
            ImGui::SameLine();
            ImGui::BeginDisabled(m_Design.shapes.empty());
            if (ImGui::Button("Score this design", ImVec2(-FLT_MIN, 30))) StartAuto(true);
            ImGui::SetItemTooltip("Measure the current design with the same score, without changing it");
            ImGui::EndDisabled();
            ImGui::EndDisabled();
            return;
        }
        const JobControl* c = m_AutoJob.Control();
        int it = 0, n = 0, accepted = 0, rounds = 0;
        if (m_AutoLive) {
            std::lock_guard<std::mutex> l(m_AutoLive->mutex);
            it = m_AutoLive->iteration;
            n = m_AutoLive->iterations;
            accepted = m_AutoLive->accepted;
            rounds = m_AutoLive->rounds;
        }
        char label[96];
        std::string stage = c ? c->Stage() : std::string();
        if (stage == "Optimizing") snprintf(label, sizeof(label), "%d / %d  (%.0f s)", it, n, GetTime() - m_AutoStart);
        else snprintf(label, sizeof(label), "%s", stage.c_str());
        ImGui::ProgressBar(c ? c->Progress() : 0.0f, ImVec2(WidthBeside("Stop"), 0), label);
        ImGui::SameLine();
        if (ImGui::Button("Stop", ImVec2(-FLT_MIN, 0))) m_AutoJob.Cancel();
        ImGui::SetItemTooltip("Stop and keep the best design found so far");
        if (rounds > 0) ImGui::TextDisabled("%d of %d rewrite rounds changed the design", accepted, rounds);
    }

    void App::GenerateScore() {
        const AutoScore& s = m_AutoScore;
        ImGui::SeparatorText("Loss");
        if (!m_AutoJob.Busy() && !(m_AutoScoreFor == m_Design)) ImGui::TextColored(Amber, "The design changed since: score it again to update.");
        if (ImGui::BeginTable("views", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("silhouettes");
            ImGui::TableSetupColumn("covered");
            ImGui::TableSetupColumn("extra");
            ImGui::TableHeadersRow();
            const char* rows[] = { "Open vs mesh", "Closed vs sphere" };
            float values[2][2] = { { s.viewCover, s.viewExtra }, { s.sphereViewCover, s.sphereViewExtra } };
            for (int r = 0; r < 2; r++) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(rows[r]);
                for (float v : values[r]) {
                    ImGui::TableNextColumn();
                    ImGui::Text("%.0f%%", v * 100.0f);
                }
            }
            ImGui::EndTable();
        }
        ImGui::SetItemTooltip("Means over the views. Covered: target pixels the design covers, as a share of the target's.\n"
                              "Extra: design pixels where the target has none, as a share of the target's.");
        ImVec4 worstColor = s.worstView < m_AutoParams.minViewCoverage ? Amber : ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled);
        ImGui::TextColored(worstColor, "Worst view %.0f%% covered", s.worstView * 100.0f);
        if (m_HasAutoViews) {
            float w = ImGui::GetContentRegionAvail().x;
            rlImGuiImageSize(&m_AutoViews, (int)w, (int)(m_AutoViews.height * w / m_AutoViews.width));
            ImGui::SetItemTooltip("Each view: open design vs the mesh (top), closed design vs the sphere (bottom).\n"
                                  "Dark: both cover the pixel. Red: only the target (missing). Blue: only the design (extra).");
        }
        ImGui::TextDisabled("loss %.3f (fit %.3f, sphere %.3f, coverage %.3f)", s.total, s.fit, s.sphere, s.coveragePenalty);
        ImGui::SetItemTooltip("Lower is better: (1 - goal) * fit + goal * sphere + coverage penalty. Fit and sphere: the\n"
                              "mean over the views of the share of the target missed plus the share of extra pixels.");
        ImGui::TextDisabled("sphere %.1f mm across", s.radius * 2.0f);
        if (m_AutoHistory.size() > 1) {
            auto [lo, hi] = std::minmax_element(m_AutoHistory.begin(), m_AutoHistory.end());
            ImGui::PlotLines("##history", m_AutoHistory.data(), (int)m_AutoHistory.size(), 0, "loss", *lo - 0.01f, fminf(*hi, *lo + 2.0f) + 0.01f,
                             ImVec2(-FLT_MIN, 60));
        }
    }

    void App::GenerateSuite() {
        ImGui::SeparatorText("Test suite");
        Hint("Runs these settings on every .obj in a folder, one after another, and renders one square video of them "
             "all: each mesh's best design over the whole search, then the result closing and opening.");
        bool busy = m_SuiteJob.Busy();
        ImGui::BeginDisabled(busy);
        ImGui::SetNextItemWidth(WidthBeside("Choose..."));
        ImGui::InputTextWithHint("##suitefolder", "folder of .obj files", m_SuiteFolder, sizeof(m_SuiteFolder));
        ImGui::SameLine();
        if (ImGui::Button("Choose...")) BrowseSuiteFolder();
        // counted again when the folder changes, and every couple of seconds (not every frame: it may be slow to list)
        if (m_SuiteCountedFolder != m_SuiteFolder || GetTime() - m_SuiteCountedAt > 2.0) {
            m_SuiteCountedFolder = m_SuiteFolder;
            m_SuiteCountedAt = GetTime();
            m_SuiteMeshCount = m_SuiteFolder[0] ? (int)SuiteMeshes(m_SuiteFolder).size() : 0;
        }
        int meshes = m_SuiteMeshCount;
        if (m_SuiteFolder[0]) ImGui::TextDisabled("%d .obj file%s; results go to its autodesign_suite folder", meshes, meshes == 1 ? "" : "s");
        SuiteParams& sp = m_SuiteParams;
        ImGui::DragFloat("Mesh size", &sp.meshSize, 1.0f, 0.0f, 1000.0f, sp.meshSize > 0.0f ? "%.0f mm" : "as is");
        ImGui::SetItemTooltip("Each mesh is centred and scaled so its longest side is this long (0 keeps the file's units)");
        ImGui::TextUnformatted("Video");
        for (int size : { 720, 1080, 1440, 2160 }) {
            ImGui::SameLine();
            char label[16];
            snprintf(label, sizeof(label), "%d##vs", size);
            if (ImGui::RadioButton(label, sp.videoSize == size)) sp.videoSize = size;
        }
        ImGui::SliderFloat("Search part", &sp.searchSeconds, 2.0f, 30.0f, "%.0f s");
        ImGui::SetItemTooltip("Video time for the search: one frame per evenly spaced snapshot of the best design");
        ImGui::SliderFloat("Each motion", &sp.motionSeconds, 0.5f, 8.0f, "%.1f s");
        ImGui::SetItemTooltip("Closing, then opening, after a short pause on the result");
        ImGui::EndDisabled();

        if (!busy) {
            ImGui::BeginDisabled(meshes == 0 || m_AutoJob.Busy());
            if (ImGui::Button("Run test suite", ImVec2(-FLT_MIN, 30))) StartSuite();
            ImGui::SetItemTooltip("Uses the settings above (always starting from two hinged boxes). Needs ffmpeg on the PATH.");
            ImGui::EndDisabled();
            if (!m_SuiteVideo.empty()) {
                if (ImGui::Button("Open video")) OpenURL(m_SuiteVideo.c_str());
                ImGui::SameLine();
                if (ImGui::Button("Open folder")) OpenURL(GetDirectoryPath(m_SuiteVideo.c_str()));
                ImGui::SameLine();
                ImGui::TextDisabled("%s", GetFileName(m_SuiteVideo.c_str()));
            }
            return;
        }
        std::string phase, name;
        int index = 0, count = 0, it = 0, n = 0;
        float progress = 0.0f;
        if (m_SuiteLive) {
            std::lock_guard<std::mutex> l(m_SuiteLive->mutex);
            phase = m_SuiteLive->phase;
            name = m_SuiteLive->name;
            index = m_SuiteLive->index;
            count = m_SuiteLive->count;
            progress = m_SuiteLive->progress;
            if (m_SuiteLive->search) {
                std::lock_guard<std::mutex> l2(m_SuiteLive->search->mutex);
                it = m_SuiteLive->search->iteration;
                n = m_SuiteLive->search->iterations;
            }
        }
        char label[160];
        if (phase == "Searching") {
            progress = count > 0 ? (index + (n > 0 ? (float)it / n : 0.0f)) / count : 0.0f;
            snprintf(label, sizeof(label), "%d / %d: %s  (%.0f s)", index + 1, count, name.c_str(), GetTime() - m_SuiteStart);
        } else {
            snprintf(label, sizeof(label), "%s  (%.0f s)", phase.empty() ? "Starting" : phase.c_str(), GetTime() - m_SuiteStart);
        }
        ImGui::ProgressBar(progress, ImVec2(WidthBeside("Stop"), 0), label);
        ImGui::SameLine();
        if (ImGui::Button("Stop##suite", ImVec2(-FLT_MIN, 0))) m_SuiteJob.Cancel();
    }
}
