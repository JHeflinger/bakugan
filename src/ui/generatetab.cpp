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
        GenerateSemantic();
        GenerateStart();
        GenerateOptimizer();
        ImGui::EndDisabled();
        GenerateGrammar();                   // changes take while it runs too
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
        Hint("A sphere is cut into pieces and hinged, so that open it looks like the mesh. Closed, the pieces are the sphere "
             "again, always: only the open pose and the motion are scored.");
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
        ImGui::SliderFloat("Collisions", &p.collisionWeight, 0.0f, 5.0f, p.collisionWeight > 0.0f ? "%.2f" : "ignored");
        ImGui::SetItemTooltip("What pieces overlapping each other costs: in the open pose, and the poses checked between\n"
                              "(together they weigh as much as the open pose). Closed, the pieces fill the sphere side by side.");
        ImGui::BeginDisabled(p.collisionWeight <= 0.0f);
        ImGui::SliderInt("Poses between", &p.motionChecks, 0, 64);
        ImGui::SetItemTooltip("Poses checked for overlaps between open and closed. They go only where some hinge is moving,\n"
                              "so no part of the motion can hide a collision from them.");
        ImGui::SliderFloat("Follow busy times", &p.motionCheckBias, 0.0f, 1.0f, "%.2f");
        ImGui::SetItemTooltip("0: the poses between are spread evenly over the times any hinge moves. 1: placed by how many\n"
                              "hinges move at once. Between, a mix: some evenly everywhere a hinge moves, the rest where most do.");
        ImGui::EndDisabled();
        ImGui::SliderFloat("Hinges on pieces", &p.hingeWeight, 0.0f, 10.0f, p.hingeWeight > 0.0f ? "%.2f" : "ignored");
        ImGui::SetItemTooltip("What a joint's pivot being away from the two pieces it joins costs: how far it is outside\n"
                              "each, beyond 1%% of the mesh's size, over the mesh's size. A real hinge sits where its pieces meet.");
        ImGui::SliderFloat("Keep the surface", &p.surfaceWeight, 0.0f, 10.0f, p.surfaceWeight > 0.0f ? "%.2f" : "ignored");
        ImGui::SetItemTooltip("What hollows carving into the sphere's surface cost (about the share of the surface carved away).\n"
                              "Hollows take material out from inside the sphere; this keeps its outside whole, closed.");
        ImGui::SeparatorText("In 3D");
        ImGui::SliderFloat("Transport", &p.transportWeight, 0.0f, 5.0f, p.transportWeight > 0.0f ? "%.2f" : "off");
        ImGui::SetItemTooltip("Optimal transport: how far the sphere's material has to travel, opened, to fill the mesh.\n"
                              "Every bit of material is paired with a bit of the mesh it should fill, however far away, so a\n"
                              "piece opened to the wrong place is pulled to where it's needed.");
        ImGui::BeginDisabled(p.transportWeight <= 0.0f);
        ImGui::SliderInt("Transport points", &p.transportPoints, 256, 4096);
        ImGui::SetItemTooltip("Points of the sphere's material, and points filling the mesh: more is finer and slower (its\n"
                              "cost grows with the square)");
        ImGui::EndDisabled();
        ImGui::SliderFloat("3D SDF", &p.sdfWeight, 0.0f, 5.0f, p.sdfWeight > 0.0f ? "%.2f" : "off");
        ImGui::SetItemTooltip("The open design's signed distance against the mesh's, at points in and around the mesh: inside\n"
                              "where the mesh is, outside where it isn't. Sees dents and hollows outlines miss.");
        ImGui::SliderFloat("Chamfer", &p.chamferWeight, 0.0f, 5.0f, p.chamferWeight > 0.0f ? "%.2f" : "off");
        ImGui::SetItemTooltip("The Chamfer distance: how far the mesh's surface is from the design's, and the design's from the\n"
                              "mesh's. Carved pieces can't follow fine surface detail, so it's best light, beside the others.");
        ImGui::BeginDisabled(p.sdfWeight <= 0.0f && p.chamferWeight <= 0.0f);
        ImGui::SliderInt("3D points", &p.shapeSamples, 300, 20000);
        ImGui::SetItemTooltip("The mesh's points for the SDF and Chamfer terms (a third on its surface)");
        ImGui::EndDisabled();
        ImGui::Checkbox("Mesh can move", &p.moveTarget);
        ImGui::SetItemTooltip("The mesh the open design is compared with may be moved, to wherever the design fits it best\n"
                              "(it stays where the views can see all of it). Off, it stays where it is.");
        ImGui::SameLine();
        ImGui::Checkbox("Match volumes", &p.scaleTarget);
        ImGui::SetItemTooltip("The mesh is scaled to the sphere's volume, so the sphere's material fills it exactly (with the\n"
                              "sphere sized from the mesh, that's its own size; set a sphere size to make the design bigger or\n"
                              "smaller). Off, it's compared at its own size. The reference mesh shows the placement found.");
    }

    // DINOv3 comparing the views as pictures (see autodesign/semantic.cpp): what it costs, the model and runtime, how the
    // views are drawn, and which embeddings are compared.
    void App::GenerateSemantic() {
        AutoParams& p = m_AutoParams;
        ImGui::SeparatorText("Semantic loss (DINOv3)");
        ImGui::SliderFloat("Semantic", &p.semanticWeight, 0.0f, 5.0f, p.semanticWeight > 0.0f ? "%.2f" : "off");
        ImGui::SetItemTooltip("What the views looking different to DINOv3 costs: each view of the open design is compared with\n"
                              "the placed mesh's");
        ImGui::SliderFloat("Silhouettes", &p.silhouetteWeight, 0.0f, 5.0f, p.silhouetteWeight > 0.0f ? "%.2f" : "off");
        ImGui::SetItemTooltip("What the outline comparison (above) costs, beside it. 0: only DINOv3 judges how the design looks.");
        if (p.semanticWeight <= 0.0f) return;

        // a path, typed or browsed for
        auto path = [&](const char* id, const char* hint, std::string& value, const char* filters, const char* tip) {
            char buf[1024];
            snprintf(buf, sizeof(buf), "%s", value.c_str());
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - ImGui::GetFrameHeight() * 2.2f);
            if (ImGui::InputTextWithHint(id, hint, buf, sizeof(buf))) value = buf;
            ImGui::SetItemTooltip("%s", tip);
            ImGui::SameLine();
            ImGui::PushID(id);
            std::string chosen;
            if (ImGui::Button("...", ImVec2(-FLT_MIN, 0)) && FileDialog(false, filters, value, &chosen)) value = chosen;
            ImGui::PopID();
        };
        path("##model", "DINOv3 model (.onnx)", p.semanticModel, "onnx",
             "The loss graph tools/autodesign/export_dinov3.py writes from DINOv3's weights. The views are drawn at the\n"
             "size it was exported for (112 by default: 7 x 7 patches; 224 gives 14 x 14, about 4x slower).");
        path("##ort", "ONNX Runtime library (empty: found as usual)", p.onnxRuntime, nullptr,
             "onnxruntime.dll / libonnxruntime.so / libonnxruntime.dylib, from ONNX Runtime's releases or a pip install of\n"
             "onnxruntime (in its capi folder). Empty: the system's usual search for that name.");
        if (p.semanticModel.empty()) Warn("Choose the model (make it with tools/autodesign/export_dinov3.py).");

        const char* looks[] = { "Silhouette", "Depth", "Shaded" };
        int look = (int)p.semanticLook;
        if (ImGui::Combo("Drawn as", &look, looks, 3)) p.semanticLook = (SemanticLook)look;
        ImGui::SetItemTooltip("How the design and the mesh are drawn for DINOv3, both alike:\n"
                              "Silhouette: white on black. Depth: nearer is brighter.\n"
                              "Shaded: a grey material under a key light and a fill light, on dark grey.");
        int mode = p.patchWeight <= 0.0f ? 0 : (p.patchWeight >= 1.0f ? 1 : 2);
        ImGui::TextUnformatted("Compare");
        ImGui::SameLine();
        if (ImGui::RadioButton("Global", mode == 0)) p.patchWeight = 0.0f;
        ImGui::SetItemTooltip("Each view's global (CLS) embedding against the target's: what the whole picture is");
        ImGui::SameLine();
        if (ImGui::RadioButton("Patches", mode == 1)) p.patchWeight = 1.0f;
        ImGui::SetItemTooltip("Each patch's embedding against the target's patch in the same place: what is where");
        ImGui::SameLine();
        if (ImGui::RadioButton("Mix", mode == 2)) p.patchWeight = 0.5f;
        if (mode == 2) {
            ImGui::SliderFloat("Patch share", &p.patchWeight, 0.01f, 0.99f, "%.2f");
            ImGui::SetItemTooltip("How much the patch comparison counts against the global one");
        }
        ImGui::SliderInt("Shade samples", &p.semanticSamples, 0, 20000);
        ImGui::SetItemTooltip("Covered pixels whose shading the gradient follows each step, drawn by how much each matters.\n"
                              "More is a steadier gradient and slower; edges are always all followed. (Silhouettes have no\n"
                              "shading to follow.)");
        ImGui::Checkbox("Use the GPU (CUDA)", &p.semanticGpu);
        ImGui::SetItemTooltip("Run DINOv3 with CUDA, if this ONNX Runtime has it (the onnxruntime-gpu builds); the CPU otherwise");
    }

    void App::GenerateStart() {
        AutoParams& p = m_AutoParams;
        ImGui::SeparatorText("Start from");
        ImGui::DragFloat("Sphere size", &p.sphereDiameter, 0.5f, 0.0f, 500.0f, p.sphereDiameter > 0.0f ? "%.1f mm across" : "auto");
        ImGui::SetItemTooltip("The sphere that's carved, at the mesh's centre of volume. Auto: the mesh's volume (the open\n"
                              "design then has the mesh's volume too: pieces only move).");
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
        ImGui::SetItemTooltip("Grammar rewrites tried in a round (drawn by the Grammar weights below)");
        ImGui::SliderInt("Lookahead", &p.lookahead, 1, 100);
        ImGui::SetItemTooltip("Adam steps every rewrite, and the design without one, get before they are compared:\n"
                              "the best of them carries on");
        ImGui::SliderFloat("Rewrite margin", &p.rewriteMargin, 0.0f, 0.05f, "%.3f");
        ImGui::SetItemTooltip("How much better (a share of the loss) a rewrite that adds something must do than leaving the design\n"
                              "alone to be kept (ones that only win by noise are clutter), and how much worse one that removes\n"
                              "something may do (a simpler design that does about as well is better)");
        ImGui::SliderInt("Max pieces", &p.maxPieces, 2, 512);
        ImGui::SetItemTooltip("Adding a cut makes one more piece: no more than this");
        ImGui::SliderInt("Max hollows", &p.maxHollows, 0, 512);
        ImGui::SetItemTooltip("Hollows: material taken out of the sphere (from every piece) without making a piece");
        int seed = (int)p.seed;
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.5f);
        if (ImGui::InputInt("Seed", &seed)) p.seed = (unsigned)std::max(seed, 0);
        ImGui::SetItemTooltip("Same seed and settings, same result");
        ImGui::SameLine();
        if (ImGui::Button("Shuffle")) p.seed = (unsigned)GetRandomValue(1, 999999);
    }

    // How often each grammar rule is drawn for a rewrite. Live: a run reads them at every rewrite round.
    void App::GenerateGrammar() {
        static const std::vector<AutoRuleInfo> rules = AutoRules();
        AutoParams& p = m_AutoParams;
        if (p.ruleWeights.size() != rules.size()) {
            p.ruleWeights.resize(rules.size());
            for (size_t i = 0; i < rules.size(); i++) p.ruleWeights[i] = rules[i].weight;
        }
        ImGui::SeparatorText("Grammar");
        Hint("How often each rewrite is drawn (0: never). These apply while it runs, from its next rewrite round.");
        bool changed = false;
        for (size_t i = 0; i < rules.size(); i++) {
            ImGui::PushID((int)i);
            changed |= ImGui::SliderFloat(rules[i].name.c_str(), &p.ruleWeights[i], 0.0f, 3.0f, "%.2f");
            p.ruleWeights[i] = std::max(p.ruleWeights[i], 0.0f);     // typed in (Ctrl+click) below the range
            ImGui::SetItemTooltip(rules[i].simplifies ? "Takes something away: kept when the design does no worse (within the rewrite margin)"
                                                      : "Adds something: kept when the design does better (by the rewrite margin)");
            ImGui::PopID();
        }
        if (ImGui::Button("Default weights")) {
            for (size_t i = 0; i < rules.size(); i++) p.ruleWeights[i] = rules[i].weight;
            changed = true;
        }
        if (changed && m_AutoLive) {
            std::lock_guard<std::mutex> l(m_AutoLive->mutex);
            m_AutoLive->ruleWeights = p.ruleWeights;
        }
    }

    // The buttons, or the progress of a run.
    void App::GenerateRun() {
        ImGui::Checkbox("Show progress live", &m_AutoPreview);
        ImGui::SetItemTooltip("Put the best design so far in the editor while the search runs");
        ImGui::SameLine();
        ImGui::Checkbox("Show sphere", &m_ShowAutoSphere);
        ImGui::SetItemTooltip("The sphere that's carved, in the output view (the design closes into it)");
        if (!m_AutoJob.Busy()) {
            ImGui::BeginDisabled(!m_HasReference || m_SuiteJob.Busy());
            if (ImGui::Button("Generate", ImVec2(HalfWidth(), 30))) StartAuto(false);
            ImGui::SetItemTooltip("Replace the design with a carved sphere (Ctrl+Z brings it back)");
            ImGui::SameLine();
            ImGui::BeginDisabled(m_Design.shapes.empty());
            if (ImGui::Button("Score this design", ImVec2(-FLT_MIN, 30))) StartAuto(true);
            ImGui::SetItemTooltip("Measure the current design with the same score (its open pose against the mesh where it is),\n"
                                  "without changing it");
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
            const char* rows[] = { "Open vs mesh" };
            float values[1][2] = { { s.viewCover, s.viewExtra } };
            for (int r = 0; r < 1; r++) {
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
            ImGui::SetItemTooltip("Each view: the open design against the mesh where it's placed.\n"
                                  "Dark: both cover the pixel. Red: only the mesh (missing). Blue: only the design (extra).");
        }
        if (m_HasAutoSemantic) {
            ImGui::TextDisabled("DINOv3: %.3f, worst view %.3f", s.semantic, s.worstSemantic);
            ImGui::SetItemTooltip("The mean over the views of DINOv3's loss against the placed mesh (0: the same; 1 - cosine of the embeddings)");
            float w = ImGui::GetContentRegionAvail().x;
            rlImGuiImageSize(&m_AutoSemantic, (int)w, (int)(m_AutoSemantic.height * w / m_AutoSemantic.width));
            ImGui::SetItemTooltip("What DINOv3 compares, per view: the placed mesh (top), the open design (bottom)");
        }
        ImVec4 clashColor = s.clashOpen + s.clashMotion > 0.005f ? Amber : ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled);
        ImGui::TextColored(clashColor, "Overlapping: %.1f%% open, %.1f%% worst between", s.clashOpen * 100.0f, s.clashMotion * 100.0f);
        ImGui::SetItemTooltip("Volume inside two pieces at once, as a share of the sphere's. Worst between: at %.0f%% of the\n"
                              "closing motion. (Closed, carved pieces never overlap.)", s.clashMotionT * 100.0f);
        GenerateTimeline();
        ImVec4 pivotColor = s.hingeGap > 0.02f * Vector3Distance(ReferenceBounds().min, ReferenceBounds().max) ? Amber
                                                                                                             : ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled);
        ImGui::TextColored(pivotColor, "Farthest pivot %.1f mm outside one of its pieces", fmaxf(s.hingeGap, 0.0f));
        ImGui::TextColored(s.surfaceCarved > 0.02f ? Amber : ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled), "Sphere surface carved: %.1f%%",
                           s.surfaceCarved * 100.0f);
        ImGui::SetItemTooltip("The share of the sphere's surface hollows reach (closed, it shows there)");
        if (m_AutoParams.transportWeight > 0.0f) {
            ImGui::TextDisabled("Transport: the material travels %.1f mm (rms); %.0f%% of the mesh unfilled", s.transportMove,
                                s.transportUnfilled * 100.0f);
            ImGui::SetItemTooltip("How far, root mean square, the open design's material is from the part of the mesh it's paired with");
        }
        if (m_AutoParams.sdfWeight > 0.0f || m_AutoParams.chamferWeight > 0.0f) {
            ImGui::TextDisabled("3D: mesh %.1f mm from the design, design %.1f mm from the mesh (on average)", s.meshGap, s.designGap);
            ImGui::SetItemTooltip("Mean distance from the mesh's surface to the open design's, and back");
        }
        ImGui::TextDisabled("loss %.3f (fit %.3f, coverage %.3f, semantic %.3f, collisions %.3f, hinges %.3f, surface %.3f, transport %.3f, "
                            "sdf %.3f, chamfer %.3f)", s.total, s.fit, s.coveragePenalty, s.semantic, s.collision, s.hinges, s.surface,
                            s.transport, s.sdf, s.chamfer);
        ImGui::SetItemTooltip("Lower is better: silhouettes * (fit + coverage penalty) + semantic * its weight\n"
                              "+ collisions * their weight + hinges * their weight + surface * its weight\n"
                              "+ transport, sdf and chamfer * theirs.\n"
                              "Fit: the mean over the views of the share of the mesh missed plus the share of extra pixels.\n"
                              "Collisions: overlap depth over its volume, against the sphere's volume.\n"
                              "Hinges: how far pivots are outside their pieces (beyond 1%% of the mesh's size), over its size.");
        ImGui::TextDisabled("sphere %.1f mm across; mesh moved %.1f mm, scaled %.2fx", s.radius * 2.0f, Vector3Length(s.target.offset), s.target.scale);
        if (m_AutoHistory.size() > 1) {
            auto [lo, hi] = std::minmax_element(m_AutoHistory.begin(), m_AutoHistory.end());
            ImGui::PlotLines("##history", m_AutoHistory.data(), (int)m_AutoHistory.size(), 0, "loss", *lo - 0.01f, fminf(*hi, *lo + 2.0f) + 0.01f,
                             ImVec2(-FLT_MIN, 60));
        }
    }

    // The closing motion as a strip: each hinge's turn as a bar, the poses checked for overlaps as ticks.
    void App::GenerateTimeline() {
        const std::vector<float>& checks = m_AutoScore.checkTimes;
        if (m_Design.hinges.empty()) return;
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 at = ImGui::GetCursorScreenPos();
        float w = ImGui::GetContentRegionAvail().x, row = 5.0f, gap = 2.0f;
        int n = (int)m_Design.hinges.size();
        float h = n * (row + gap) + 10.0f;
        ImGui::InvisibleButton("##timeline", ImVec2(w, h));
        ImU32 track = ImGui::GetColorU32(ImGuiCol_FrameBg), bar = ImGui::GetColorU32(ImGuiCol_PlotHistogram), tick = ImGui::GetColorU32(ImGuiCol_Text);
        for (int i = 0; i < n; i++) {
            const Hinge& hg = m_Design.hinges[i];
            float y = at.y + i * (row + gap);
            dl->AddRectFilled(ImVec2(at.x, y), ImVec2(at.x + w, y + row), track);
            dl->AddRectFilled(ImVec2(at.x + w * hg.t0, y), ImVec2(at.x + w * hg.t1, y + row), bar);
        }
        for (float t : checks) dl->AddLine(ImVec2(at.x + w * t, at.y + h - 8.0f), ImVec2(at.x + w * t, at.y + h), tick);
        ImGui::SetItemTooltip("The closing motion, open (left) to closed (right): a bar for when each hinge turns, a tick for\n"
                              "each pose checked for overlaps between (%d)", (int)checks.size());
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
