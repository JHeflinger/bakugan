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
        Hint("Builds a design from the reference mesh: superfrustum bodies fitted to it in the open pose, hinged so they "
             "close into a sphere. Shapes, cuts, groups and hinges are all tuned together against one score.");
        GenerateInput();
        ImGui::BeginDisabled(m_AutoJob.Busy());
        GenerateGoal();
        GenerateStart();
        GenerateSpace();
        GenerateSearch();
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
        ImGui::TextUnformatted("Score by");
        ImGui::SameLine();
        if (ImGui::RadioButton("Silhouettes", p.loss == AutoLoss::Silhouettes)) p.loss = AutoLoss::Silhouettes;
        ImGui::SetItemTooltip("Compare outlines from several directions: the open design's against the mesh's, the closed\n"
                              "design's against a disc of the sphere's size. Outlines can't see depth or hidden hollows:\n"
                              "more views pin the shape down better.");
        ImGui::SameLine();
        if (ImGui::RadioButton("Volume", p.loss == AutoLoss::Volume)) p.loss = AutoLoss::Volume;
        ImGui::SetItemTooltip("Compare filled volumes and surfaces in 3D");
        if (p.loss == AutoLoss::Silhouettes) {
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
        ImGui::SetItemTooltip("Diameter the closed design should have. Auto: a sphere of the closed design's own volume.");
    }

    void App::GenerateStart() {
        AutoParams& p = m_AutoParams;
        ImGui::SeparatorText("Start from");
        if (ImGui::RadioButton("The mesh", !p.fromCurrent)) p.fromCurrent = false;
        ImGui::SetItemTooltip("Split the mesh into pieces (k-means), one box-like superfrustum and one body each, hinged in a tree");
        ImGui::SameLine();
        ImGui::BeginDisabled(m_Design.shapes.empty());
        if (ImGui::RadioButton("The current design", p.fromCurrent)) p.fromCurrent = true;
        ImGui::SetItemTooltip("Refine what is in the editor (its hinges keep their hardware settings)");
        ImGui::EndDisabled();
        if (!p.fromCurrent) {
            ImGui::SliderInt("Pieces", &p.seedBodies, 1, std::max(1, p.maxBodies));
            ImGui::SetItemTooltip("Bodies to begin with. The search adds, splits, merges and removes bodies from there.");
        }
    }

    void App::GenerateSpace() {
        AutoParams& p = m_AutoParams;
        ImGui::SeparatorText("Design space");
        ImGui::SliderInt("Min bodies", &p.minBodies, 1, 12);
        ImGui::SetItemTooltip("Hinged parts (top level groups) the design must keep. Merging everything into one body\n"
                              "dodges every collision, so without a floor the search tends to stop folding.");
        ImGui::SliderInt("Max bodies", &p.maxBodies, 1, 12);
        ImGui::SetItemTooltip("Hinged parts (top level groups)");
        if (p.maxBodies < p.minBodies) p.maxBodies = p.minBodies;
        ImGui::SliderInt("Max shapes", &p.maxShapes, 1, 48);
        ImGui::SetItemTooltip("Superfrustums in all, cuts included");
        ImGui::Checkbox("Cuts", &p.allowCuts);
        ImGui::SetItemTooltip("Shapes and features may remove material");
        ImGui::SameLine();
        ImGui::Checkbox("Blended features", &p.allowFeatures);
        ImGui::SetItemTooltip("Shapes may be grouped into features inside a body, smoothly merged with a blend");
        ImGui::Checkbox("Hinge hardware", &p.hingeHardware);
        ImGui::SetItemTooltip("Give the result's hinges knuckles and leaves. Off, they are bare pivots: nothing is added\n"
                              "to the bodies. The score never includes hardware either way.");
        ImGui::SliderFloat("Collisions", &p.collisionWeight, 0.0f, 5.0f, p.collisionWeight > 0.0f ? "%.2f" : "ignored");
        ImGui::SetItemTooltip("What bodies overlapping each other costs, in the open and in the closed pose (and on the way\n"
                              "between with motion checks). It starts low and reaches this by 60%% of the run, so folds can\n"
                              "form first.");
        ImGui::BeginDisabled(p.collisionWeight <= 0.0f);
        ImGui::SliderInt("Motion checks", &p.motionChecks, 0, 32, p.motionChecks > 0 ? "%d moments" : "ends only");
        ImGui::SetItemTooltip("Moments between open and closed also checked for bodies overlapping, evenly spaced and eased\n"
                              "like the animation (the opening motion's too, for hinges with their own opening timing).\n"
                              "Together they weigh as much as one end pose. More catch quicker swings, and cost more.");
        ImGui::BeginDisabled(p.motionChecks <= 0);
        ImGui::Checkbox("Hinge timing", &p.allowTiming);
        ImGui::EndDisabled();
        ImGui::SetItemTooltip("Let the search change when each hinge turns: slide and stretch its share of the closing\n"
                              "motion, move it before or after another hinge's, give it its own opening timing. Only the\n"
                              "motion checks can see timing, so it needs them.");
        ImGui::EndDisabled();
        ImGui::SliderFloat("Simplicity", &p.simplicity, 0.0f, 1.0f, "%.2f");
        ImGui::SetItemTooltip("What each shape and group costs in the score: higher keeps the design leaner");
    }

    void App::GenerateSearch() {
        AutoParams& p = m_AutoParams;
        ImGui::SeparatorText("Search");
        ImGui::SliderInt("Scorings", &p.iterations, 500, 60000, "%d", ImGuiSliderFlags_Logarithmic);
        ImGui::SetItemTooltip("How many designs the search may score (each in both poses): its budget");
        ImGui::SliderInt("Rewrites per round", &p.rewritesPerRound, 1, 64);
        ImGui::SetItemTooltip("Stochastic Rewrite Descent: grammar rewrites drawn and tried in each round. Every one that\n"
                              "improves the score is applied, all together (skipping any that conflict).");
        ImGui::SliderInt("Steps per round", &p.stepsPerRound, 0, 100);
        ImGui::SetItemTooltip("Descent steps on the shapes', hinges' and blends' values between rewrite rounds");
        const char* qualities[] = { "Fast", "Normal", "Fine" };
        int q = std::clamp(p.quality, 0, 2);
        ImGui::TextUnformatted("Samples");
        for (int k = 0; k < 3; k++) {
            ImGui::SameLine();
            if (ImGui::RadioButton(qualities[k], q == k)) p.quality = k;
        }
        ImGui::SetItemTooltip("Points the score is measured at: finer is steadier and slower");
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
        if (stage == "Searching") snprintf(label, sizeof(label), "%d / %d  (%.0f s)", it, n, GetTime() - m_AutoStart);
        else snprintf(label, sizeof(label), "%s", stage.c_str());
        ImGui::ProgressBar(c ? c->Progress() : 0.0f, ImVec2(WidthBeside("Stop"), 0), label);
        ImGui::SameLine();
        if (ImGui::Button("Stop", ImVec2(-FLT_MIN, 0))) m_AutoJob.Cancel();
        ImGui::SetItemTooltip("Stop and keep the best design found so far");
        if (it > 0) ImGui::TextDisabled("%d rewrites applied in %d rounds", accepted, rounds);
    }

    void App::GenerateScore() {
        const AutoScore& s = m_AutoScore;
        ImGui::SeparatorText("Score");
        if (!m_AutoJob.Busy() && !(m_AutoScoreFor == m_Design)) ImGui::TextColored(Amber, "The design changed since: score it again to update.");
        // a table of percentages: a heading row, then one row per pose
        auto table = [](const char* id, std::initializer_list<const char*> heads, std::initializer_list<float> open,
                        std::initializer_list<float> closed) {
            if (!ImGui::BeginTable(id, (int)heads.size(), ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) return;
            for (const char* h : heads) ImGui::TableSetupColumn(h);
            ImGui::TableHeadersRow();
            const char* rows[] = { "Open vs mesh", "Closed vs sphere" };
            const std::initializer_list<float>* values[] = { &open, &closed };
            for (int r = 0; r < 2; r++) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(rows[r]);
                for (float v : *values[r]) {
                    ImGui::TableNextColumn();
                    ImGui::Text("%.0f%%", v * 100.0f);
                }
            }
            ImGui::EndTable();
        };
        if (s.silhouettes) {
            table("views", { "outlines", "covered", "extra" }, { s.viewCover, s.viewExtra }, { s.sphereViewCover, s.sphereViewExtra });
            ImGui::SetItemTooltip("Means over the views. Covered: target pixels the design covers, as a share of the target's.\n"
                                  "Extra: design pixels where the target has none, as a share of the target's.");
            ImVec4 worstColor = s.worstView < m_AutoParams.minViewCoverage ? Amber : ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled);
            ImGui::TextColored(worstColor, "Worst view %.0f%% covered", s.worstView * 100.0f);
            if (s.offFrame > 0.005f) {
                ImGui::SameLine();
                ImGui::TextDisabled("  %.0f%% of the material outside the views", s.offFrame * 100.0f);
            }
            if (m_HasAutoViews) {
                float w = ImGui::GetContentRegionAvail().x;
                rlImGuiImageSize(&m_AutoViews, (int)w, (int)(m_AutoViews.height * w / m_AutoViews.width));
                ImGui::SetItemTooltip("Each view: open design vs the mesh (top), closed design vs the sphere (bottom).\n"
                                      "Dark: both cover the pixel. Red: only the target (missing). Blue: only the design (extra).");
            }
        } else {
            table("score", { "", "covered", "outside", "surface gap" }, { s.fitCover, s.fitOutside, s.fitGap },
                  { s.sphereIoU, s.sphereSpill, s.sphereGap });
            ImGui::SetItemTooltip("Open: share of the mesh's volume the design fills, share of the design's volume outside the\n"
                                  "mesh, and how far the mesh surface is from the design's (share of 8%% of the mesh's size).\n"
                                  "Closed: overlap with the sphere (shared over combined volume), how far material reaches\n"
                                  "outside it on average (share of its radius), and the surface gap (share of 30%% of the radius).");
        }
        ImVec4 clashColor = s.clashOpen + s.clashClosed + s.clashMotion > 0.005f ? Amber : ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled);
        if (m_AutoParams.motionChecks > 0) {
            ImGui::TextColored(clashColor, "Bodies overlapping: %.1f%% open, %.1f%% closed, %.1f%% on the way", s.clashOpen * 100.0f,
                               s.clashClosed * 100.0f, s.clashMotion * 100.0f);
            ImGui::SetItemTooltip("Share of the design's volume inside two bodies at once. On the way: the worst checked moment\n"
                                  "(%.0f%% of the %s)", s.clashMotionT * 100.0f, s.clashMotionOpening ? "opening" : "closing");
        } else {
            ImGui::TextColored(clashColor, "Bodies overlapping: %.1f%% open, %.1f%% closed", s.clashOpen * 100.0f, s.clashClosed * 100.0f);
            ImGui::SetItemTooltip("Share of the design's volume inside two bodies at once (the positions between are not checked)");
        }
        ImGui::TextDisabled("score %.3f (fit %.3f, sphere %.3f, collisions %.3f, costs %.3f)", s.total, s.fit, s.sphere, s.collision, s.penalty);
        ImGui::SetItemTooltip("Lower is better: (1 - goal) * fit + goal * sphere + collisions * their weight + costs.\n"
                              "Fit: missing and extra volume against the mesh (both as shares of its volume), how far extra\n"
                              "material reaches, the surface gap. Costs: per shape and group (Simplicity), pivots off their\n"
                              "bodies, bodies on no hinge.");
        ImGui::TextDisabled("sphere %.1f mm across", s.radius * 2.0f);
        if (m_AutoHistory.size() > 1) {
            auto [lo, hi] = std::minmax_element(m_AutoHistory.begin(), m_AutoHistory.end());
            ImGui::PlotLines("##history", m_AutoHistory.data(), (int)m_AutoHistory.size(), 0, "best score", *lo - 0.01f, *hi + 0.01f,
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
            ImGui::SetItemTooltip("Uses the settings above (always starting from each mesh). Needs ffmpeg on the PATH.");
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
