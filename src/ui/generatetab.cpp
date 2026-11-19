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
        GenerateFinish();
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
            ImGui::SliderInt("Views", &p.views, 6, 128);
            ImGui::SetItemTooltip("Directions the outlines are compared from, spread over a half sphere (from the other\n"
                                  "half they would only be mirror images)");
            ImGui::SliderInt("View pixels", &p.viewResolution, 64, 4000);
            ImGui::SetItemTooltip("Pixels across each view: finer is more exact and slower");
            ImGui::SliderFloat("Min coverage", &p.minViewCoverage, 0.0f, 1.0f, "%.2f");
            ImGui::SetItemTooltip("Share of every view's target outline the design must cover. Each view short of it costs\n"
                                  "steeply (10 per whole share missing), so no single view can be given up.");
        }
        ImGui::SliderFloat("Collisions", &p.collisionWeight, 0.0f, 5.0f, p.collisionWeight > 0.0f ? "%.2f" : "ignored");
        ImGui::SetItemTooltip("What pieces overlapping each other costs: in the open pose, and in every hinge's turn.\n"
                              "Closed, the pieces fill the sphere side by side.");
        ImGui::BeginDisabled(p.collisionWeight <= 0.0f);
        ImGui::SliderFloat("... by the end", &p.collisionRamp, 1.0f, 100.0f, "x %.0f", ImGuiSliderFlags_Logarithmic);
        ImGui::SetItemTooltip("Overlaps cost this many times as much by the end of the search as at its start, rising steadily:\n"
                              "the search may try things early on, and has to have the pieces clear of each other when it ends.");
        ImGui::SliderInt("Checks per hinge", &p.hingeChecks, 6, 64);
        ImGui::SetItemTooltip("Poses checked for overlaps in every hinge's turn, spread evenly over its angle, however long or\n"
                              "short the turn is: turning faster hides nothing. Each step looks between the last step's poses.");
        bool order = p.orderHinges || p.dependencies;
        ImGui::BeginDisabled(p.dependencies);
        if (ImGui::Checkbox("Open one at a time, the freest first", &order)) p.orderHinges = order;
        ImGui::EndDisabled();
        ImGui::SetItemTooltip("The hinges open one after another, each in its own share of the motion. From closed, the hinge\n"
                              "whose turn runs into the least opens next, so one that needs another out of its way waits for\n"
                              "it (the order is looked at again at every rewrite round). Off: each hinge's timing is a number\n"
                              "Adam tunes, and hinges may turn at once.");
        ImGui::EndDisabled();
        ImGui::Checkbox("Hinges hold each other shut", &p.dependencies);
        ImGui::SetItemTooltip("Every hinge but the first to open should be held closed by a piece that moves away before its\n"
                              "turn: then one hinge keeps the whole design shut, and letting it go lets the rest unfurl in turn.\n"
                              "The search keeps such pieces next to each other; when it is done, small ledges are cut from the\n"
                              "sphere's surface: each is part of the holding piece and lies over the piece it holds.");
        if (p.dependencies) {
            ImGui::SliderFloat("Holding", &p.dependencyWeight, 0.0f, 5.0f, "%.2f");
            ImGui::SetItemTooltip("What a hinge with nothing near enough to hold it costs: how far the nearest piece that could is,\n"
                                  "along the sphere's surface, over the sphere's radius");
            ImGui::DragFloat("Ledge size", &p.ledges.size, 0.05f, 0.0f, 10.0f, p.ledges.size > 0.0f ? "%.2f mm" : "auto");
            ImGui::SetItemTooltip("How far a ledge reaches over the piece it holds (auto: 12%% of the sphere's radius). It is 2.5\n"
                                  "times as wide, 0.6 times as thick, and anchored as far again in its own piece.");
            ImGui::DragFloat("Ledge reach", &p.ledges.reach, 0.1f, 0.0f, 50.0f, p.ledges.reach > 0.0f ? "%.1f mm" : "auto");
            ImGui::SetItemTooltip("How much of another piece's surface a ledge may cross to reach the piece it holds, as a strap\n"
                                  "set into a groove (auto: 35%% of the sphere's radius)");
        }
        ImGui::SliderFloat("Hinges on pieces", &p.hingeWeight, 0.0f, 10.0f, p.hingeWeight > 0.0f ? "%.2f" : "ignored");
        ImGui::SetItemTooltip("What a hinge being away from the two pieces it joins costs: how far each is from its axis, along\n"
                              "the length a hinge's barrel takes up there, beyond half of what a leaf can bridge (Finishing: Hinge reach).\n"
                              "A real hinge sits where its pieces meet. It is also what limits the search most: with it ignored,\n"
                              "axes go wherever the pieces swing clear of each other, which no hinge could be built on.");
        ImGui::SliderFloat("Keep the surface", &p.surfaceWeight, 0.0f, 10.0f, p.surfaceWeight > 0.0f ? "%.2f" : "ignored");
        ImGui::SetItemTooltip("What hollows carving into the sphere's surface cost (about the share of the surface carved away).\n"
                              "Hollows take material out from inside the sphere; this keeps its outside whole, closed.");
        ImGui::Checkbox("Mesh can move", &p.moveTarget);
        ImGui::SetItemTooltip("The mesh the open design is compared with may be moved, to wherever the design fits it best\n"
                              "(it stays where the views can see all of it). Off, it stays where it is.");
        ImGui::SameLine();
        ImGui::Checkbox("Mesh can scale", &p.scaleTarget);
        ImGui::SetItemTooltip("... and scaled about its centre of volume (0.3x .. 3.3x). The reference mesh shows the\n"
                              "placement found, with the design it was found for.");
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
        int from = (int)m_AutoStartFrom;
        if (ImGui::RadioButton("The sphere", from == 0)) from = 0;
        ImGui::SetItemTooltip("The sphere, whole: one piece, cut up from there");
        ImGui::SameLine();
        if (ImGui::RadioButton("This design", from == 1)) from = 1;
        ImGui::SetItemTooltip("Carry the search on from the design in the editor (one the generator made: a result, or a\n"
                              "loaded one). What finishing added (hinge hardware, ledges, carving, a latch) is left out\n"
                              "and done again at the end, as the Finishing settings say.");
        ImGui::SameLine();
        if (ImGui::RadioButton("A design file", from == 2)) from = 2;
        ImGui::SetItemTooltip("Carry the search on from a saved design (.txt): a result the generator saved, the command\n"
                              "line's <name>.txt or <name>.partial.txt, or one saved from here after generating");
        m_AutoStartFrom = (AutoStartFrom)from;
        if (m_AutoStartFrom == AutoStartFrom::File) {
            char buf[1024];
            snprintf(buf, sizeof(buf), "%s", m_AutoStartPath.c_str());
            ImGui::SetNextItemWidth(WidthBeside("..."));
            if (ImGui::InputTextWithHint("##startdesign", "design to start from (.txt)", buf, sizeof(buf))) m_AutoStartPath = buf;
            ImGui::SetItemTooltip("The search starts from this design's pieces, cuts and hinges. It is moved and scaled onto this\n"
                                  "run's sphere; the mesh starts where the file says it was placed for it (when it says).");
            ImGui::SameLine();
            std::string chosen;
            if (ImGui::Button("...##startdesign") && FileDialog(false, "txt", m_AutoStartPath.empty() ? std::string(m_DesignPath) : m_AutoStartPath, &chosen))
                m_AutoStartPath = chosen;
        }
        if (m_AutoStartFrom != AutoStartFrom::Sphere)
            Hint("Overlaps start cheap again (Collisions: ... by the end), as in any run: lower that, or the steps, to refine gently.");
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
        ImGui::BeginDisabled(!m_Gpu->Ok());
        ImGui::Checkbox("Draw the views on the GPU", &m_AutoUseGpu);
        ImGui::EndDisabled();
        ImGui::SetItemTooltip("%s", m_Gpu->Ok() ? "Every step's views of the open design (the silhouettes, and the pictures DINOv3 is shown) are\n"
                                                  "raymarched by the graphics card instead of the CPU's cores. The comparison with the mesh, the\n"
                                                  "shading of the semantic pictures and the gradient stay on the CPU; DINOv3 itself runs where\n"
                                                  "its own CUDA option says. The editor's window is drawn with the same card: it redraws only a\n"
                                                  "dozen times a second while a search runs, between the search's steps."
                                                : m_Gpu->Error().c_str());
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

    // What makes a carved sphere printable (data/finish.hpp): at the end of a search, or on the design as it is.
    void App::GenerateFinish() {
        AutoParams& p = m_AutoParams;
        ImGui::SeparatorText("Finishing");
        bool busy = m_AutoJob.Busy() || m_SuiteJob.Busy() || m_FinishJob.Busy();
        ImGui::BeginDisabled(busy);
        ImGui::Checkbox("Build the hinges when done", &p.buildHinges);
        ImGui::SetItemTooltip("Give every hinge the smallest barrel that prints: three knuckles on a pin, where both of its pieces\n"
                              "are along the axis. The search then keeps each axis where such a hinge fits, and (under the\n"
                              "surface) checks collisions with the axis where it will be.");
        ImGui::Checkbox("Hinges under the surface", &p.hinges.flush);
        ImGui::SetItemTooltip("Each axis far enough under the sphere's surface for its barrel to be inside the sphere, so the closed\n"
                              "sphere stays round. Off: the barrel is on the axis wherever that is, and sticks out if it is at\n"
                              "the surface. On an existing design this moves the axes inward: carve collisions away afterwards.");
        ImGui::SliderFloat("Hinge size", &p.hinges.scale, 0.75f, 2.5f, "x %.2f");
        {
            HingeSizes sz = PlanHinge(0.0f, p.hinges);
            ImGui::SetItemTooltip("Now: a %.1f mm pin in a %.1f mm barrel, %.1f mm long", 2.0f * sz.pinRadius, 2.0f * sz.radius, sz.length);
        }
        ImGui::SliderFloat("Hinge reach", &p.hinges.reach, 0.0f, 8.0f, p.hinges.reach > 0.0f ? "%.1f mm" : "smallest");
        ImGui::SetItemTooltip("How far a piece may be from a hinge's axis and still be joined to it, by a leaf that long. The search\n"
                              "counts a hinge as on its pieces within this ('Hinges on pieces'), so more reach leaves it freer\n"
                              "where it puts the axes, at the price of longer leaves.");
        ImGui::DragFloat("Hinge clearance", &p.hinges.clearance, 0.01f, 0.1f, 1.0f, "%.2f mm");
        ImGui::SetItemTooltip("Between the pin and the knuckles around it, and between knuckles (what your printer needs between\n"
                              "parts that must not fuse)");
        {
            int pin = p.hinges.pin == PinStyle::Fused ? 0 : 1;
            if (ImGui::Combo("Hinge pin", &pin, "Printed in place\0Separate pin\0")) p.hinges.pin = pin == 0 ? PinStyle::Fused : PinStyle::Separate;
            ImGui::SetItemTooltip("Printed in place: the pin is part of one piece and runs through the other's knuckle.\n"
                                  "Separate: both are bored for a pin pushed in afterwards (under the surface, through a channel\n"
                                  "cut from the pin's end out of the sphere).");
        }
        ImGui::Checkbox("Carve collisions away when done", &p.carveWhenDone);
        ImGui::SetItemTooltip("Wherever two pieces still run into each other between open and closed, one of them is cut by\n"
                              "everything the other passes through on its way (its whole sweep, not samples of it): the one that\n"
                              "loses less, counting what shows on the closed sphere eight times. No shapes are added: each body\n"
                              "gets a clearance, listed under it in the outliner.");
        ImGui::DragFloat("Carve clearance", &p.carve.clearance, 0.01f, 0.0f, 3.0f, "%.2f mm");
        ImGui::SetItemTooltip("Room left where carved surfaces move past each other (none where they just touch, closed)");
        ImGui::SliderInt("Carve checks", &p.carve.checksPerHinge, 8, 96);
        ImGui::SetItemTooltip("Poses looked at in every hinge's turn to find where pieces meet. The cut itself follows the whole\n"
                              "motion between them; afterwards it checks again at twice as many poses, and carves anything left");
        ImGui::Checkbox("Latch the first hinge when done", &p.latchWhenDone);
        ImGui::SetItemTooltip("A press latch on the hinge that opens first (with 'Hinges hold each other shut', the one that keeps\n"
                              "the whole design closed): a small rocker in the sphere's surface, flush with it. Its catch lies\n"
                              "over a lip on the lid's edge; pressing its other end in steps the catch back off the lip.");
        ImGui::SliderFloat("Latch size", &p.latch.scale, 0.6f, 1.6f, "x %.2f");
        ImGui::DragFloat("Latch clearance", &p.latch.clearance, 0.01f, 0.15f, 1.0f, "%.2f mm");
        ImGui::SetItemTooltip("The gap printed around the rocker (what your printer needs between parts that must not fuse)");
        ImGui::EndDisabled();
        if (m_FinishJob.Busy()) {
            DrawJob(m_FinishJob.Control(), true, "finish", [&] { m_FinishJob.Cancel(); });
            return;
        }
        ImGui::BeginDisabled(busy || m_Design.hinges.empty());
        float third = (ImGui::GetContentRegionAvail().x - 3.0f * ImGui::GetStyle().ItemSpacing.x) / 4.0f;
        if (ImGui::Button("Add hinges", ImVec2(third, 0))) StartFinish(Finish::Hinges);
        ImGui::SetItemTooltip("Build the selected hinge's hardware, or every hinge's when none is selected (any design; Ctrl+Z undoes it)");
        ImGui::SameLine();
        if (ImGui::Button("Carve now", ImVec2(third, 0))) StartFinish(Finish::Carve);
        ImGui::SetItemTooltip("Carve the current design's collisions away (any design; Ctrl+Z undoes it)");
        ImGui::SameLine();
        if (ImGui::Button("Add ledges", ImVec2(third, 0))) StartFinish(Finish::Ledges);
        ImGui::SetItemTooltip("Make the current design's hinges hold each other shut, in the order they open (a carved sphere)");
        ImGui::SameLine();
        if (ImGui::Button("Add latch", ImVec2(-FLT_MIN, 0))) StartFinish(Finish::Latch);
        ImGui::SetItemTooltip("A press latch on the selected hinge, or on the first to open when none is selected (a carved sphere)");
        ImGui::EndDisabled();
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
            if (ImGui::Button(m_AutoStartFrom == AutoStartFrom::Sphere ? "Generate" : "Continue", ImVec2(HalfWidth(), 30))) StartAuto(false);
            ImGui::SetItemTooltip("%s", m_AutoStartFrom == AutoStartFrom::Sphere
                                            ? "Replace the design with a carved sphere (Ctrl+Z brings it back)"
                                            : "Carry the search on from the design chosen in Start from, and replace the design with\n"
                                              "the result (Ctrl+Z brings it back)");
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
        if (m_AutoUseGpu && m_Gpu->Drawn() > 0)
            ImGui::TextDisabled("GPU: %d drawings, %.1f ms the last, %d shaders compiled", m_Gpu->Drawn(), m_Gpu->LastMs(), m_Gpu->Compiles());
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
        ImGui::SetItemTooltip("Volume inside two pieces at once, as a share of the sphere's, at the poses the search last looked\n"
                              "at. Worst between: at %.0f%% of the closing motion. (Closed, carved pieces never overlap.)", s.clashMotionT * 100.0f);
        if (m_HasAutoChecked && s.radius > 0.0f) {
            const MotionOverlap& m = m_AutoChecked;
            const double sphere = 4.0 / 3.0 * PI * pow(s.radius, 3.0);
            bool clear = m.openVolume + m.worstVolume < 0.0005 * sphere;
            ImGui::TextColored(clear ? Green : Amber, "Checked at %d poses: %.2f%% open, %.2f%% worst between, %.2f mm deep", m.poses,
                               100.0 * m.openVolume / sphere, 100.0 * m.worstVolume / sphere, m.deepest);
            ImGui::SetItemTooltip("The finished design, checked once more at 24 poses in every hinge's turn on a finer grid.\n"
                                  "What is left can be carved away (Finishing).");
        }
        if (s.dependents > 0) {
            ImGui::TextColored(s.holdable < s.dependents ? Amber : ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled),
                               "Holding: %d of %d hinges have a piece near enough to hold them", s.holdable, s.dependents);
            ImGui::SetItemTooltip("Of the hinges after the first to open, those with a piece within a ledge's reach that moves away\n"
                                  "before their turn (the status line says which ledges were made)");
        }
        GenerateTimeline();
        ImVec4 pivotColor = s.hingeGap > 0.02f * Vector3Distance(ReferenceBounds().min, ReferenceBounds().max) ? Amber
                                                                                                             : ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled);
        ImGui::TextColored(pivotColor, "Farthest hinge %.1f mm off one of its pieces", fmaxf(s.hingeGap, 0.0f));
        ImGui::TextColored(s.surfaceCarved > 0.02f ? Amber : ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled), "Sphere surface carved: %.1f%%",
                           s.surfaceCarved * 100.0f);
        ImGui::SetItemTooltip("The share of the sphere's surface hollows reach (closed, it shows there)");
        ImGui::TextDisabled("loss %.3f (fit %.3f, coverage %.3f, semantic %.3f, collisions %.3f x %.0f, hinges %.3f, surface %.3f, holding %.3f)",
                            s.total, s.fit, s.coveragePenalty, s.semantic, s.collision, s.collisionScale, s.hinges, s.surface, s.dependency);
        ImGui::SetItemTooltip("Lower is better: silhouettes * (fit + coverage penalty) + semantic * its weight\n"
                              "+ collisions * their weight + hinges * their weight + surface * its weight.\n"
                              "Fit: the mean over the views of the share of the mesh missed plus the share of extra pixels.\n"
                              "Collisions: overlap depth over its volume, against the sphere's volume (times how far the\n"
                              "search had raised their cost when this was scored).\n"
                              "Hinges: how far each hinge's pieces are from its axis, along the length its barrel takes up, over the mesh's size.");
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
