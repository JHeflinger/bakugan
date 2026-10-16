#include "core/app.hpp"
#include "util/utils.hpp"
#include <imgui_internal.h>
#include <algorithm>
#include <raymath.h>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <functional>

namespace Bakugan {
    static bool ColorEdit(const char* id, Color* c) {
        float f[3] = { c->r / 255.0f, c->g / 255.0f, c->b / 255.0f };
        if (!ImGui::ColorEdit3(id, f, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel)) return false;
        *c = { (unsigned char)(f[0] * 255), (unsigned char)(f[1] * 255), (unsigned char)(f[2] * 255), 255 };
        return true;
    }

    static void Swatch(Color c) {
        ImVec2 p = ImGui::GetCursorScreenPos();
        float s = ImGui::GetTextLineHeight();
        ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(p.x, p.y + 1), ImVec2(p.x + s, p.y + s - 1), IM_COL32(c.r, c.g, c.b, 255), 3.0f);
        ImGui::Dummy(ImVec2(s, s));
    }

    // A progress bar coloured from good (green) to bad (red).
    static void Meter(const char* label, float value, bool higherIsBetter, const char* fmt, const char* tip) {
        float good = higherIsBetter ? value : 1.0f - value;
        good = std::clamp(good, 0.0f, 1.0f);
        ImVec4 col = ImVec4(0.95f - 0.7f * good, 0.3f + 0.55f * good, 0.3f, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, col);
        char buf[64];
        snprintf(buf, sizeof(buf), fmt, value * 100.0f);
        ImGui::ProgressBar(std::clamp(value, 0.0f, 1.0f), ImVec2(ImGui::GetContentRegionAvail().x * 0.55f, 0), buf);
        ImGui::PopStyleColor();
        ImGui::SameLine();
        ImGui::TextUnformatted(label);
        if (tip) ImGui::SetItemTooltip("%s", tip);
    }

    bool App::DrawJob(const JobControl* c, bool busy, const char* id) {
        if (!busy || !c) return false;
        ImGui::PushID(id);
        std::string stage = c->Stage();
        ImGui::ProgressBar(c->Progress(), ImVec2(ImGui::GetContentRegionAvail().x - 60, 0), stage.c_str());
        ImGui::SameLine();
        if (ImGui::Button("Stop", ImVec2(-FLT_MIN, 0))) {
            if (m_DecomposeJob.Busy()) m_DecomposeJob.Cancel();
            if (m_FoldJob.Busy()) m_FoldJob.Cancel();
            if (m_RestoreJob.Busy()) m_RestoreJob.Cancel();
            m_RunAll = false;
        }
        ImGui::SetItemTooltip("Stop and keep the best result so far");
        ImGui::PopID();
        return true;
    }

    void App::DrawLibrary() {
        if (!ImGui::CollapsingHeader("Model library", ImGuiTreeNodeFlags_DefaultOpen)) return;
        float w = ImGui::GetContentRegionAvail().x;
        if (ImGui::Button("Open model...", ImVec2(w * 0.4f, 0))) BrowseModel();
        ImGui::SetItemTooltip("Pick an .obj .gltf .glb .iqm .vox or .m3d file (or a design .txt).\n"
                              "You can also drop files onto a panel. Loading resets the design.");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-ImGui::CalcTextSize("Load").x - ImGui::GetStyle().FramePadding.x * 2 - ImGui::GetStyle().ItemSpacing.x);
        bool enter = ImGui::InputTextWithHint("##path", "or type a path", m_OpenPath, sizeof(m_OpenPath),
                                              ImGuiInputTextFlags_EnterReturnsTrue);
        ImGui::SameLine();
        if ((ImGui::Button("Load") || enter) && m_OpenPath[0]) OpenModelFile(m_OpenPath);
        if (ImGui::BeginChild("library", ImVec2(0, 92), ImGuiChildFlags_Borders)) {
            for (int i = 0; i < (int)m_Library.size(); i++) {
                ImGui::PushID(i);
                bool current = m_HasInput && m_Library[i].name == m_InputName;
                if (ImGui::Selectable(m_Library[i].name.c_str(), current) && !current) LoadAsset(i);
                if (!m_Library[i].path.empty()) ImGui::SetItemTooltip("%s", m_Library[i].path.c_str());
                if (ImGui::BeginDragDropSource()) {
                    ImGui::SetDragDropPayload("ASSET", &i, sizeof(int));
                    ImGui::Text("Load %s", m_Library[i].name.c_str());
                    ImGui::EndDragDropSource();
                }
                ImGui::PopID();
            }
        }
        ImGui::EndChild();
        if (m_Shape && m_Shape->ok)
            ImGui::TextDisabled("Volume %.3f   equal-volume sphere r = %.3f", m_Shape->volume, m_Shape->equalRadius);
    }

    void App::DrawEditBar() {
        ImGui::BeginDisabled(m_UndoStack.empty() || Busy());
        if (ImGui::Button("Undo")) Undo();
        ImGui::EndDisabled();
        ImGui::SetItemTooltip("Cmd/Ctrl+Z  (%d steps)", (int)m_UndoStack.size());
        ImGui::SameLine();
        ImGui::BeginDisabled(m_RedoStack.empty() || Busy());
        if (ImGui::Button("Redo")) Redo();
        ImGui::EndDisabled();
        ImGui::SetItemTooltip("Cmd/Ctrl+Shift+Z or Ctrl+Y  (%d steps)", (int)m_RedoStack.size());
        ImGui::SameLine();
        const Design& d = Shown();
        ImGui::TextDisabled("%d parts, %d hinges", (int)d.parts.size(), d.CountHinges());
    }

    void App::DrawSidebar(float width, float height) {
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(width, height));
        ImGui::Begin("Bakugan Everything", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                                    ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings);
        DrawLibrary();
        DrawEditBar();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.75f, 0.85f, 1.0f, 1.0f));
        ImGui::TextWrapped("%s", m_Status.c_str());
        ImGui::PopStyleColor();
        if (ImGui::BeginTabBar("mode")) {
            if (ImGui::BeginTabItem("Automatic")) {
                m_Mode = Mode::Automatic;
                DrawAutomatic();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Manual design")) {
                m_Mode = Mode::Manual;
                DrawManual();
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        ImGui::End();
    }

    // ---- automatic -------------------------------------------------------------------------------------------------

    void App::DrawAutomatic() {
        bool ready = m_Shape && m_Shape->ok;
        ImGui::BeginDisabled(!ready || Busy());
        if (ImGui::Button("Run everything", ImVec2(-FLT_MIN, 30))) RunAll();
        ImGui::EndDisabled();
        ImGui::SetItemTooltip("Decompose, build the hinge tree, optimise the fold, then restore the detail.");

        ImGui::SeparatorText("1  Superfrustum decomposition");
        ImGui::BeginDisabled(Busy());
        ImGui::SliderInt("Resolution", &m_DecomposeParams.resolution, 24, 96, "%d voxels");
        ImGui::SetItemTooltip("Voxels along the longest side of the model for the fitting.");
        ImGui::SliderInt("Max parts", &m_DecomposeParams.maxParts, 1, 32);
        ImGui::SliderInt("Rounds", &m_DecomposeParams.rounds, 1, 8);
        ImGui::SetItemTooltip("Residual rounds: seed in the unexplained region, fit everything, prune.");
        ImGui::SliderInt("Steps per round", &m_DecomposeParams.iterations, 10, 200);
        ImGui::SliderFloat("Parsimony", &m_DecomposeParams.parsimony, 0.0f, 0.005f, "%.4f");
        ImGui::SetItemTooltip("Price of one more primitive. Higher keeps fewer, larger parts.");
        ImGui::BeginDisabled(!ready);
        if (ImGui::Button("Decompose", ImVec2(-FLT_MIN, 0))) StartDecompose(false, -1);
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        DrawJob(m_DecomposeJob.Control(), m_DecomposeJob.Busy(), "decompose");

        ImGui::SeparatorText("2  Fold into a sphere");
        DrawFoldSettings(false);
        DrawMetrics();

        ImGui::SeparatorText("3  Restore detail");
        DrawRestorePanel();
        DrawFilesPanel();
    }

    void App::DrawFoldSettings(bool manual) {
        bool ready = m_Shape && m_Shape->ok;
        FoldParams& p = m_FoldParams;
        ImGui::BeginDisabled(Busy());
        float leeway = p.volumeTolerance * 100.0f;
        if (ImGui::SliderFloat("Size leeway", &leeway, 0.0f, 60.0f, "+/- %.0f%% radius")) p.volumeTolerance = leeway / 100.0f;
        ImGui::SetItemTooltip("How far the ball's radius may stray from the sphere with the model's volume.");
        if (ready)
            ImGui::TextDisabled("radius %.3f .. %.3f  (equal volume %.3f)", m_FoldModel.RadiusMin(), m_FoldModel.RadiusMax(),
                                m_Shape->equalRadius);
        ImGui::SliderAngle("Max hinge fold", &p.maxHingeAngle, 10.0f, 180.0f);
        if (!manual || ImGui::TreeNode("Optimiser")) {
            ImGui::SliderInt("Iterations", &p.iterations, 10, 800);
            ImGui::SliderInt("Rewrites (K)", &p.candidates, 2, 64);
            ImGui::SetItemTooltip("Grammar rewrites proposed per phase. Each is settled with a local step and kept\n"
                                  "only if it lowers the objective; compatible winners are applied together.");
            ImGui::SliderInt("Rewrite every", &p.rewriteEvery, 1, 30, "%d steps");
            ImGui::Checkbox("Grammar rewrites", &p.restructure);
            ImGui::SetItemTooltip("Reparent, split, merge, lock/unlock, axis snap, flip and reseat hinges.");
            ImGui::SameLine();
            ImGui::Checkbox("Allow splits", &p.allowSplit);
            ImGui::SetItemTooltip("Let the grammar cut long parts in two with a new hinge.");
            ImGui::Checkbox("Check the folding path", &p.checkPath);
            ImGui::SetItemTooltip("Also penalise parts passing through each other half way through the fold.");
            if (manual) ImGui::TreePop();
        }
        if (ImGui::TreeNode("Objective weights")) {
            ImGui::SliderFloat("Shell gaps", &p.wCoverage, 0.0f, 8.0f, "%.2f");
            ImGui::SetItemTooltip("Sphere surface not covered by any part.");
            ImGui::SliderFloat("Shell doubled", &p.wShellOverlap, 0.0f, 4.0f, "%.2f");
            ImGui::SliderFloat("Trimmed", &p.wTrim, 0.0f, 8.0f, "%.2f");
            ImGui::SetItemTooltip("Material that ends up outside the sphere when closed.");
            ImGui::SliderFloat("Detail", &p.wDetail, 0.0f, 6.0f, "%.2f");
            ImGui::SetItemTooltip("Extra weight on trimmed surface detail (creases, small features).");
            ImGui::SliderFloat("Collision", &p.wCollision, 0.0f, 8.0f, "%.2f");
            ImGui::SetItemTooltip("Material of different parts in the same place when closed.");
            ImGui::SliderFloat("Path collision", &p.wPath, 0.0f, 4.0f, "%.2f");
            ImGui::SliderFloat("Pivot contact", &p.wContact, 0.0f, 4.0f, "%.2f");
            ImGui::SliderFloat("Joints inside", &p.wLink, 0.0f, 4.0f, "%.2f");
            ImGui::SetItemTooltip("Keep every hinge inside the closed sphere, so no link of a chain is cut away.");
            ImGui::SliderFloat("Size prior", &p.wVolume, 0.0f, 4.0f, "%.2f");
            ImGui::SetItemTooltip("Pull towards the equal-volume sphere inside the leeway band.");
            ImGui::SliderFloat("Hinge cost", &p.hingeCost, 0.0f, 0.05f, "%.3f");
            ImGui::SetItemTooltip("Simplicity: price of each working spring hinge.");
            if (ImGui::Button("Defaults")) {
                FoldParams def;
                p.wCoverage = def.wCoverage; p.wShellOverlap = def.wShellOverlap; p.wTrim = def.wTrim; p.wDetail = def.wDetail;
                p.wCollision = def.wCollision; p.wPath = def.wPath; p.wContact = def.wContact; p.wVolume = def.wVolume; p.wLink = def.wLink;
                p.hingeCost = def.hingeCost;
            }
            ImGui::TreePop();
        }
        ImGui::BeginDisabled(!ready || Shown().parts.empty());
        float half = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
        if (ImGui::Button("Build hinge tree", ImVec2(half, 0))) BuildTree();
        ImGui::SetItemTooltip("Replace the hinges with a spanning tree over the parts' contacts (the biggest part is\n"
                              "the root), and aim each hinge to tuck its limb into the sphere.");
        ImGui::SameLine();
        static bool keepTree = false;
        if (ImGui::Button(manual ? "Refine with optimiser" : "Optimise fold", ImVec2(-FLT_MIN, 0))) StartFold(manual && keepTree);
        ImGui::SetItemTooltip("Stochastic rewrite descent over the fold grammar, from the current design.");
        if (manual) {
            ImGui::Checkbox("Keep my hinge tree", &keepTree);
            ImGui::SetItemTooltip("Only tune angles, axes, pivots and the sphere, and lock/unlock hinges.\n"
                                  "No reparenting, splitting or merging.");
        }
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        DrawJob(m_FoldJob.Control(), m_FoldJob.Busy(), "fold");
        std::vector<float> hist = m_FoldJob.Busy() && m_FoldJob.Control() ? m_FoldJob.Control()->History() : m_LastHistory;
        if (hist.size() > 1) {
            char overlay[48];
            snprintf(overlay, sizeof(overlay), "objective %.4f", hist.back());
            ImGui::PlotLines("##loss", hist.data(), (int)hist.size(), 0, overlay, FLT_MAX, FLT_MAX, ImVec2(-FLT_MIN, 54));
        }
    }

    void App::DrawMetrics() {
        if (!m_AnalysisValid) {
            ImGui::TextDisabled("Add or decompose superfrustums to see how well they fold.");
            return;
        }
        const FoldMetrics& m = m_Metrics;
        Meter("shell covered", m.coverage, true, "%.0f%%", "Share of the sphere's surface covered by a part's superfrustum when closed.");
        Meter("detail kept", 1.0f - m.trimmed, true, "%.0f%%", "Share of the (curvature weighted) material that stays inside the sphere.");
        Meter("colliding", m.collision, false, "%.1f%%", "Material samples that share space with another part when closed.");
        if (m_FoldParams.checkPath) Meter("colliding mid-fold", m.pathCollision, false, "%.1f%%", "Collisions half way through folding.");
        Meter("shell doubled", m.shellOverlap, false, "%.0f%%", "Sphere surface claimed by two parts.");
        ImGui::TextDisabled("radius %.3f (x%.2f of equal volume)   pivot misfit %.2f   objective %.4f", m.radius, m.radiusRatio,
                            m.contact, m.loss);
    }

    void App::DrawRestorePanel() {
        RestoreParams& p = m_RestoreParams;
        ImGui::BeginDisabled(Busy());
        ImGui::SliderInt("Detail resolution", &p.resolution, 32, 200, "%d voxels / diameter");
        float shell = p.shellThickness * 100.0f;
        if (ImGui::SliderFloat("Shell wall", &shell, 2.0f, 40.0f, "%.0f%% of radius")) p.shellThickness = shell / 100.0f;
        ImGui::SetItemTooltip("Thickness of the outer sphere skin each part contributes.");
        ImGui::Checkbox("Fill shell gaps", &p.fillGaps);
        ImGui::SetItemTooltip("Let each part's shell plate reach past its superfrustum to close small gaps in the sphere.");
        ImGui::SameLine();
        ImGui::BeginDisabled(!p.fillGaps);
        float gap = p.gapFill * 100.0f;
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::SliderFloat("##gap", &gap, 0.0f, 40.0f, "reach %.0f%%")) p.gapFill = gap / 100.0f;
        ImGui::EndDisabled();
        ImGui::SliderFloat("Simplify", &p.adaptivity, 0.0f, 1.0f, "%.2f");
        ImGui::SetItemTooltip("OpenVDB mesh adaptivity (0 = full resolution).");
        ImGui::BeginDisabled(m_Design.parts.empty() || !m_Mesh);
        if (ImGui::Button("Build pieces", ImVec2(-FLT_MIN, 0))) StartRestore();
        ImGui::SetItemTooltip("Cut the sphere shell from the superfrustums and give every other surface back\n"
                              "the original detail (OpenVDB level sets).");
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        DrawJob(m_RestoreJob.Control(), m_RestoreJob.Busy(), "restore");
        if (!m_Restored.empty()) {
            if (RestoredCurrent()) ImGui::TextColored(ImVec4(0.4f, 0.85f, 0.5f, 1), "%d pieces built for this design.", (int)m_Restored.size());
            else ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1), "Pieces are out of date: the design changed.");
            ImGui::Checkbox("Show pieces", &m_ShowPieces);
            ImGui::SameLine();
            ImGui::Checkbox("...in the figure too", &m_ShowPiecesInFigure);
        }
    }

    void App::DrawFilesPanel() {
        if (!ImGui::CollapsingHeader("Files")) return;
        ImGui::InputText("Export prefix", m_ExportPath, sizeof(m_ExportPath));
        float half0 = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
        if (ImGui::Button("Export OBJ + hinge sheet", ImVec2(half0, 0))) ExportParts();
        ImGui::SetItemTooltip("Writes <prefix>_open.obj (figure), <prefix>_closed.obj (ball) and <prefix>_hinges.txt.\n"
                              "Uses the built pieces, or the superfrustums when none are built.");
        ImGui::SameLine();
        if (ImGui::Button("Export as...", ImVec2(-FLT_MIN, 0))) BrowseExport();
        ImGui::InputText("Design file", m_DesignPath, sizeof(m_DesignPath));
        float half = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
        if (ImGui::Button("Save design", ImVec2(half, 0))) SaveDesign();
        ImGui::SetItemTooltip("Cmd/Ctrl+S. Superfrustums, hinges and the sphere as text.");
        ImGui::SameLine();
        ImGui::BeginDisabled(Busy());
        if (ImGui::Button("Load design", ImVec2(-FLT_MIN, 0))) LoadDesign();
        ImGui::EndDisabled();
        if (ImGui::Button("Save as...", ImVec2(half, 0))) BrowseDesign(true);
        ImGui::SameLine();
        ImGui::BeginDisabled(Busy());
        if (ImGui::Button("Open design...", ImVec2(-FLT_MIN, 0))) BrowseDesign(false);
        ImGui::EndDisabled();
    }

    // ---- manual ----------------------------------------------------------------------------------------------------

    void App::DrawManual() {
        bool locked = Busy();
        ImGui::BeginDisabled(locked);
        DrawPartList();
        DrawPartInspector();
        DrawHingeInspector();
        DrawSphereInspector();
        ImGui::EndDisabled();
        ImGui::SeparatorText("Fold analysis (live)");
        DrawMetrics();
        ImGui::SeparatorText("Assist");
        DrawFoldSettings(true);
        ImGui::SeparatorText("Restore detail");
        DrawRestorePanel();
        DrawFilesPanel();
    }

    void App::DrawPartList() {
        ImGui::SeparatorText("Superfrustums");
        float third = (ImGui::GetContentRegionAvail().x - 2 * ImGui::GetStyle().ItemSpacing.x) / 3.0f;
        if (ImGui::Button("Add", ImVec2(third, 0))) {
            SuperFrustum sf;
            sf.position = m_Shape ? m_Shape->centroid : Vector3{ 0, 1, 0 };
            if (m_Selected >= 0) sf.position = Vector3Add(m_Design.parts[m_Selected].shape.position, { 0.15f, 0.15f, 0 });
            sf.size = { 0.3f, 0.3f, 0.3f };
            Select(AddPart(sf));
        }
        ImGui::SetItemTooltip("Shift+A. New parts start rigidly attached to the selection.");
        ImGui::SameLine();
        ImGui::BeginDisabled(m_Selected < 0);
        if (ImGui::Button("Duplicate", ImVec2(third, 0))) DuplicateSelected();
        ImGui::SetItemTooltip("Shift+D");
        ImGui::SameLine();
        if (ImGui::Button("Delete", ImVec2(-FLT_MIN, 0))) DeleteSelected();
        ImGui::SetItemTooltip("Del. Children move up to the deleted part's parent.");
        ImGui::EndDisabled();
        ImGui::BeginDisabled(!m_Mesh || m_Design.parts.empty());
        float half = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
        ImGui::BeginDisabled(m_Selected < 0);
        if (ImGui::Button("Fit selected to mesh", ImVec2(half, 0))) StartDecompose(true, m_Selected);
        ImGui::EndDisabled();
        ImGui::SetItemTooltip("Gradient-fit the selected superfrustum to the part of the mesh around it.");
        ImGui::SameLine();
        if (ImGui::Button("Fit all", ImVec2(-FLT_MIN, 0))) StartDecompose(true, -1);
        ImGui::SetItemTooltip("Fit every superfrustum together (they share the mesh).");
        ImGui::EndDisabled();
        DrawJob(m_DecomposeJob.Control(), m_DecomposeJob.Busy(), "fit");

        const Design& d = m_Design;
        int action = 0;   // applied after the tree, which reads the design
        if (ImGui::BeginChild("parts", ImVec2(0, 150), ImGuiChildFlags_Borders | ImGuiChildFlags_ResizeY)) {
            if (d.parts.empty()) ImGui::TextDisabled("No parts. Add one, or decompose\nin the Automatic tab.");
            std::function<void(int)> node = [&](int i) {
                const Part& p = d.parts[i];
                std::vector<int> kids = d.Children(i);
                ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanAvailWidth;
                if (kids.empty()) flags |= ImGuiTreeNodeFlags_Leaf;
                if (i == m_Selected) flags |= ImGuiTreeNodeFlags_Selected;
                ImGui::PushID(p.id);
                Swatch(p.color);
                ImGui::SameLine();
                const char* kind = p.hinge.parent < 0 ? (i == d.Root() ? "  (root)" : "  (loose)")
                                 : p.hinge.locked ? "  rigid" : "";
                char label[96];
                if (p.hinge.parent >= 0 && !p.hinge.locked)
                    snprintf(label, sizeof(label), "%s   %.0f deg", p.name.c_str(), d.FoldAngle(i) * RAD2DEG);
                else
                    snprintf(label, sizeof(label), "%s%s", p.name.c_str(), kind);
                bool open = ImGui::TreeNodeEx("node", flags, "%s", label);
                if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) Select(i);
                if (ImGui::BeginPopupContextItem()) {
                    Select(i);
                    if (ImGui::MenuItem("Make root", nullptr, false, i != d.Root() && p.hinge.parent >= 0)) action = 1;
                    if (ImGui::MenuItem("Detach (loose)", nullptr, false, p.hinge.parent >= 0)) action = 2;
                    if (ImGui::MenuItem("Auto-aim hinge", nullptr, false, p.hinge.parent >= 0)) action = 3;
                    if (ImGui::MenuItem("Delete")) action = 4;
                    ImGui::EndPopup();
                }
                if (open) {
                    for (int k : kids) node(k);
                    ImGui::TreePop();
                }
                ImGui::PopID();
            };
            for (int i = 0; i < (int)d.parts.size(); i++)
                if (d.parts[i].hinge.parent < 0) node(i);
        }
        ImGui::EndChild();
        if (action == 1 && m_Selected >= 0) { m_Design.Reroot(m_Selected); Select(0); }
        if (action == 2 && m_Selected >= 0) m_Design.parts[m_Selected].hinge.parent = -1;
        if (action == 3) AimSelected();
        if (action == 4) DeleteSelected();
    }

    void App::DrawPartInspector() {
        if (m_Selected < 0 || m_Selected >= (int)m_Design.parts.size()) {
            ImGui::TextDisabled("Select a part (click it in a view or the list).");
            return;
        }
        Part& p = m_Design.parts[m_Selected];
        SuperFrustum& f = p.shape;
        ImGui::SeparatorText("Superfrustum");
        ColorEdit("##col", &p.color);
        ImGui::SameLine();
        char name[64];
        snprintf(name, sizeof(name), "%s", p.name.c_str());
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::InputText("##name", name, sizeof(name))) p.name = name;

        ImGui::DragFloat3("Position", &f.position.x, 0.005f, 0, 0, "%.3f");
        if (m_EulerPart != p.id || !QuaternionEquals(m_EulerQuaternion, f.rotation)) {
            m_EulerDegrees = Utils::QuaternionToEulerDegrees(f.rotation);
            m_EulerPart = p.id;
            m_EulerQuaternion = f.rotation;
        }
        if (ImGui::DragFloat3("Rotation", &m_EulerDegrees.x, 0.5f, -180, 180, "%.1f deg")) {
            f.rotation = Utils::EulerDegreesToQuaternion(m_EulerDegrees);
            m_EulerQuaternion = f.rotation;
        }
        ImGui::SetItemTooltip("The superfrustum sweeps along its local Z axis (blue gizmo arrow).");
        if (ImGui::DragFloat3("Size", &f.size.x, 0.004f, 0.01f, 10.0f, "%.3f")) f.Repair(0.01f);
        ImGui::SetItemTooltip("x, y: cross section   z: height along the axis");
        ImGui::SliderFloat("Rounding", &f.rounding, 0.0f, 1.0f, "%.2f");
        ImGui::SetItemTooltip("0 = box cross section, 1 = round (cylinder / cone / capsule)");
        ImGui::SliderFloat("Taper", &f.taper, 0.0f, 1.0f, "%.2f");
        ImGui::SetItemTooltip("1 = prism, 0 = the top shrinks to a point or ridge");
        ImGui::SliderFloat("Bulge", &f.bulge, -1.0f, 1.0f, "%.2f");
        ImGui::SetItemTooltip("Bends the axis into an arc");
        ImGui::SliderFloat("Dilation", &f.dilation, 0.0f, 1.0f, "%.3f");
        ImGui::SetItemTooltip("Grows the solid outward, rounding every edge");
        ImGui::SliderFloat("Onion", &f.onion, 0.0f, 0.95f, "%.2f");
        ImGui::SetItemTooltip("0 = solid, otherwise a hollow shell");
        f.Repair(0.01f);
    }

    void App::DrawHingeInspector() {
        if (m_Selected < 0 || m_Selected >= (int)m_Design.parts.size()) return;
        Design& d = m_Design;
        Part& p = d.parts[m_Selected];
        Hinge& h = p.hinge;
        ImGui::SeparatorText("Hinge to parent");
        const char* current = h.parent >= 0 ? d.parts[h.parent].name.c_str() : (m_Selected == d.Root() ? "(root)" : "(loose)");
        if (ImGui::BeginCombo("Parent", current)) {
            if (ImGui::Selectable("(none)", h.parent < 0)) h.parent = -1;
            for (int j = 0; j < (int)d.parts.size(); j++) {
                if (j == m_Selected || d.IsDescendant(j, m_Selected)) continue;
                ImGui::PushID(j);
                if (ImGui::Selectable(d.parts[j].name.c_str(), h.parent == j)) {
                    bool fresh = h.parent < 0;
                    if (d.Reparent(m_Selected, j) && fresh) SnapPivotToContact(m_Selected);
                }
                ImGui::PopID();
            }
            ImGui::EndCombo();
        }
        if (h.parent < 0) {
            ImGui::TextDisabled(m_Selected == d.Root() ? "The root stays still; everything folds onto it."
                                                       : "Loose parts stay still like the root. Pick a parent.");
            if (m_Selected != d.Root() && ImGui::Button("Make this the root")) { d.Reroot(m_Selected); Select(0); }
            return;
        }
        bool rigid = h.locked;
        if (ImGui::Checkbox("Rigid (glued, no hinge)", &rigid)) { h.locked = rigid; if (rigid) h.angle = 0.0f; }
        ImGui::BeginDisabled(h.locked);
        ImGui::DragFloat3("Pivot", &h.pivot.x, 0.004f, 0, 0, "%.3f");
        ImGui::SetItemTooltip("A point on the hinge line, in the open pose.");
        if (ImGui::DragFloat3("Axis", &h.axis.x, 0.01f, -1.0f, 1.0f, "%.2f")) {
            float len = Vector3Length(h.axis);
            h.axis = len > 1e-4f ? Vector3Scale(h.axis, 1.0f / len) : Vector3{ 1, 0, 0 };
        }
        ImGui::SetItemTooltip("The part folds closed by rotating about +axis (right-hand rule).");
        float maxA = fminf(h.maxAngle, m_FoldParams.maxHingeAngle);
        ImGui::SliderAngle("Fold angle", &h.angle, 0.0f, maxA * RAD2DEG);
        ImGui::SetItemTooltip("Closed angle. The spring opens it back to 0; it can only turn one way.\n"
                              "Drag the ring in the Ball view to turn it there.");
        ImGui::SliderAngle("Stop", &h.maxAngle, 0.0f, 180.0f);
        ImGui::SetItemTooltip("Mechanical stop: the furthest this hinge can fold.");
        h.angle = std::clamp(h.angle, 0.0f, h.maxAngle);
        float third = (ImGui::GetContentRegionAvail().x - 2 * ImGui::GetStyle().ItemSpacing.x) / 3.0f;
        if (ImGui::Button("Snap pivot", ImVec2(third, 0))) SnapPivotToContact(m_Selected);
        ImGui::SetItemTooltip("Move the pivot to the middle of where this part touches its parent.");
        ImGui::SameLine();
        if (ImGui::Button("Auto-aim", ImVec2(third, 0))) AimSelected();
        ImGui::SetItemTooltip("Search the axis and angle that best tuck this part into the sphere.");
        ImGui::SameLine();
        if (ImGui::Button("Flip", ImVec2(-FLT_MIN, 0))) h.axis = Vector3Negate(h.axis);
        ImGui::SetItemTooltip("Fold the other way round.");
        ImGui::EndDisabled();
    }

    void App::DrawSphereInspector() {
        ImGui::SeparatorText("Target sphere");
        Ball& b = m_Design.ball;
        ImGui::DragFloat3("Centre", &b.center.x, 0.004f, 0, 0, "%.3f");
        ImGui::SetItemTooltip("In the root part's frame (the root does not move while folding).");
        if (m_FoldModel.Valid()) {
            ImGui::SliderFloat("Radius", &b.radius, m_FoldModel.RadiusMin(), m_FoldModel.RadiusMax(), "%.3f");
            b.radius = std::clamp(b.radius, m_FoldModel.RadiusMin(), m_FoldModel.RadiusMax());
            ImGui::SetItemTooltip("Limited by the size leeway around the equal-volume radius.");
            if (ImGui::Button("Equal volume")) b.radius = m_FoldModel.EqualRadius();
            ImGui::SameLine();
            if (ImGui::Button("Centre on mass")) b.center = m_FoldModel.Centroid();
        }
    }
}
