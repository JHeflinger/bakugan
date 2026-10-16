#include "app.hpp"
#include "util/utils.hpp"
#include <imgui_internal.h>
#include <rlImGui.h>
#include <tinyfiledialogs.h>
#include <ImGuizmo.h>
#include <algorithm>
#include <raymath.h>
#include <raylib.h>
#include <imgui.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>

#define MAX_UNDO 200

namespace Bakugan {
    App::App() {
        SetTraceLogLevel(LOG_NONE);
        SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_WINDOW_HIGHDPI | FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
        InitWindow(1280, 720, "Bakugan Everything");
        int mon = GetCurrentMonitor();
        int mw = GetMonitorWidth(mon), mh = GetMonitorHeight(mon);
        int w = std::min(1700, (int)(mw * 0.92f)), h = std::min(980, (int)(mh * 0.88f));
        SetWindowSize(w, h);
        SetWindowPosition((mw - w) / 2, (mh - h) / 2);
        SetExitKey(KEY_NULL);
        SetTargetFPS(60);
        rlImGuiSetup(true);
        ImGui::GetIO().IniFilename = nullptr;
        m_Shader = LoadShaderFromMemory(Utils::GetVertexShader(), Utils::GetFragmentShader());
        m_ViewPositionLocation = GetShaderLocation(m_Shader, "viewPos");
        AddToLibrary({ "Critter (demo)", "", Primitive::Critter });
        AddToLibrary({ "Dragon (demo)", "", Primitive::Dragon });
        AddToLibrary({ "Cube", "", Primitive::Cube });
        AddToLibrary({ "Sphere", "", Primitive::Sphere });
        AddToLibrary({ "Cylinder", "", Primitive::Cylinder });
        AddToLibrary({ "Torus", "", Primitive::Torus });
        AddToLibrary({ "Knot", "", Primitive::Knot });
        LoadAsset(0);
        m_UndoStack.clear();
        m_Committed = Capture();
    }

    App::~App() {
        m_DecomposeJob.Stop();
        m_FoldJob.Stop();
        m_RestoreJob.Stop();
        ClearRestored();
        for (auto& [id, pm] : m_PartModels)
            if (pm.loaded) UnloadModel(pm.model);
        if (m_HasInput) UnloadModel(m_Input);
        m_Figure.Unload();
        m_Ball.Unload();
        UnloadShader(m_Shader);
        rlImGuiShutdown();
        CloseWindow();
    }

    void App::Run() {
        while (!WindowShouldClose()) Frame();
    }

    // ---- history ---------------------------------------------------------------------------------------------------

    EditState App::Capture() const {
        return { m_Design, m_Selected };
    }

    void App::Restore(const EditState& st) {
        m_Design = st.design;
        m_Selected = st.selected < (int)m_Design.parts.size() ? st.selected : -1;
    }

    bool App::Interacting() {
        return ImGui::IsAnyItemActive() || ImGuizmo::IsUsingAny() || ImGui::IsMouseDown(ImGuiMouseButton_Left);
    }

    void App::TrackHistory() {
        if (Interacting()) return;
        if (m_Committed.design == m_Design) {
            m_Committed.selected = m_Selected;
            return;
        }
        m_UndoStack.push_back(std::move(m_Committed));
        if (m_UndoStack.size() > MAX_UNDO) m_UndoStack.erase(m_UndoStack.begin());
        m_RedoStack.clear();
        m_Committed = Capture();
    }

    void App::Undo() {
        if (Busy()) return;
        TrackHistory();
        if (m_UndoStack.empty() || Interacting()) return;
        m_RedoStack.push_back(Capture());
        m_Committed = std::move(m_UndoStack.back());
        m_UndoStack.pop_back();
        Restore(m_Committed);
    }

    void App::Redo() {
        if (Busy()) return;
        TrackHistory();
        if (m_RedoStack.empty() || Interacting()) return;
        m_UndoStack.push_back(Capture());
        m_Committed = std::move(m_RedoStack.back());
        m_RedoStack.pop_back();
        Restore(m_Committed);
    }

    void App::HandleKeys() {
        ImGuiIO& io = ImGui::GetIO();
        if (io.WantTextInput) return;
        bool cmd = io.KeyCtrl || io.KeySuper;
        if (cmd) {
            if (ImGui::IsKeyPressed(ImGuiKey_Z)) { if (io.KeyShift) Redo(); else Undo(); }
            if (ImGui::IsKeyPressed(ImGuiKey_Y)) Redo();
            if (ImGui::IsKeyPressed(ImGuiKey_S)) SaveDesign();
            return;
        }
        bool overView = m_Figure.Hovered() || m_Ball.Hovered();
        if (!overView) return;
        if (ImGui::IsKeyPressed(ImGuiKey_F)) {
            if (m_Figure.Hovered()) m_Figure.Frame(m_InputBounds);
            if (m_Ball.Hovered()) m_Ball.Frame(m_InputBounds);
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Space)) {
            m_FoldAnim = m_Fold > 0.5f ? 1 : 2;
            m_FoldVelocity = 0.0f;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_W)) m_GizmoOp = GizmoOp::Translate;
        if (ImGui::IsKeyPressed(ImGuiKey_E)) m_GizmoOp = GizmoOp::Rotate;
        if (ImGui::IsKeyPressed(ImGuiKey_R)) m_GizmoOp = GizmoOp::Scale;
        if (ImGui::IsKeyPressed(ImGuiKey_Q))
            m_Target = m_Target == EditTarget::Part ? EditTarget::Hinge : m_Target == EditTarget::Hinge ? EditTarget::Sphere : EditTarget::Part;
        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) Select(-1);
        if (m_Mode == Mode::Manual && !Busy()) {
            if (ImGui::IsKeyPressed(ImGuiKey_Delete) || ImGui::IsKeyPressed(ImGuiKey_Backspace)) DeleteSelected();
            if (ImGui::IsKeyPressed(ImGuiKey_D) && io.KeyShift) DuplicateSelected();
            if (ImGui::IsKeyPressed(ImGuiKey_A) && io.KeyShift) {
                SuperFrustum sf;
                sf.position = m_Shape ? m_Shape->centroid : Vector3{ 0, 1, 0 };
                sf.size = { 0.3f, 0.3f, 0.3f };
                Select(AddPart(sf));
            }
        }
    }

    // ---- library / input -------------------------------------------------------------------------------------------

    int App::AddToLibrary(const Asset& a) {
        for (int i = 0; i < (int)m_Library.size(); i++)
            if (!a.path.empty() && m_Library[i].path == a.path) return i;
        m_Library.push_back(a);
        return (int)m_Library.size() - 1;
    }

    void App::LoadAsset(int index) {
        if (Busy()) { m_Status = "Wait for the running job to finish (or stop it) before loading a model."; return; }
        const Asset& a = m_Library[index];
        Model m;
        std::string err;
        if (!Utils::LoadAsset(a, &m, &err)) { m_Status = err; return; }
        SetInputModel(m, a.name);
    }

    bool App::OpenModelFile(const std::string& path) {
        if (path.empty()) return false;
        if (!FileExists(path.c_str())) { m_Status = "File not found: " + path; return false; }
        if (IsFileExtension(path.c_str(), ".txt")) {
            snprintf(m_DesignPath, sizeof(m_DesignPath), "%s", path.c_str());
            LoadDesign();
            return true;
        }
        if (!Utils::IsModelFile(path.c_str())) { m_Status = "Unsupported file type: " + path; return false; }
        LoadAsset(AddToLibrary({ GetFileName(path.c_str()), path }));
        return true;
    }

    // Native file dialogs (tinyfiledialogs: zenity / kdialog / yad on Linux, the system dialogs elsewhere). They work
    // where drag and drop doesn't, e.g. Wayland compositors talking to this XWayland window.
    static const char* NoDialog = "No file chosen. (On Linux the file dialog needs zenity, kdialog or yad installed; you can also type a path.)";

    void App::BrowseModel() {
        if (Busy()) { m_Status = "Wait for the running job to finish (or stop it) before loading a model."; return; }
        const char* filters[] = { "*.obj", "*.gltf", "*.glb", "*.iqm", "*.vox", "*.m3d" };
        const char* path = tinyfd_openFileDialog("Open a model", "", 6, filters, "3D models", 0);
        if (!path) { m_Status = NoDialog; return; }
        snprintf(m_OpenPath, sizeof(m_OpenPath), "%s", path);
        OpenModelFile(path);
    }

    void App::BrowseDesign(bool save) {
        const char* filters[] = { "*.txt" };
        const char* path = save ? tinyfd_saveFileDialog("Save design", m_DesignPath, 1, filters, "Bakugan designs")
                                : tinyfd_openFileDialog("Load design", m_DesignPath, 1, filters, "Bakugan designs", 0);
        if (!path) { m_Status = NoDialog; return; }
        snprintf(m_DesignPath, sizeof(m_DesignPath), "%s", path);
        if (save) SaveDesign();
        else LoadDesign();
    }

    void App::BrowseExport() {
        const char* filters[] = { "*.obj" };
        const char* path = tinyfd_saveFileDialog("Export prefix (writes _open.obj, _closed.obj, _hinges.txt)", m_ExportPath, 1, filters, "OBJ");
        if (!path) { m_Status = NoDialog; return; }
        std::string p = path;
        for (const char* suffix : { "_open.obj", "_closed.obj", ".obj" })
            if (p.size() > strlen(suffix) && p.compare(p.size() - strlen(suffix), strlen(suffix), suffix) == 0) {
                p.resize(p.size() - strlen(suffix));
                break;
            }
        snprintf(m_ExportPath, sizeof(m_ExportPath), "%s", p.c_str());
        ExportParts();
    }

    void App::SetInputModel(Model m, const std::string& name) {
        if (m_HasInput) UnloadModel(m_Input);
        m_Input = m;
        m_HasInput = true;
        m_InputName = name;
        for (int i = 0; i < m_Input.materialCount; i++) m_Input.materials[i].shader = m_Shader;

        // normalise: longest side 2 units, standing on the ground, centred
        BoundingBox b = GetModelBoundingBox(m_Input);
        Vector3 size = Vector3Subtract(b.max, b.min);
        float extent = fmaxf(size.x, fmaxf(size.y, size.z));
        float s = extent > 1e-6f ? 2.0f / extent : 1.0f;
        Vector3 off = { -(b.min.x + b.max.x) * 0.5f * s, -b.min.y * s, -(b.min.z + b.max.z) * 0.5f * s };
        m_InputTransform = MatrixMultiply(MatrixScale(s, s, s), MatrixTranslate(off.x, off.y, off.z));
        m_Input.transform = m_InputTransform;
        TriMesh mesh = Utils::ModelToTriMesh(m_Input, m_InputTransform);
        m_InputBounds = mesh.box;
        m_Mesh = std::make_shared<const TriMesh>(std::move(mesh));
        m_Shape = std::make_shared<const ShapeAnalysis>(AnalyzeShape(*m_Mesh, 64, 4000));

        Design d;
        if (m_Shape->ok) {
            d.ball.center = m_Shape->centroid;
            d.ball.radius = m_Shape->equalRadius;
        }
        m_Design = d;
        m_Selected = -1;
        m_UndoStack.clear();
        m_RedoStack.clear();
        m_Committed = Capture();
        ClearRestored();
        m_LastHistory.clear();
        m_FoldModel = FoldModel(m_Shape, m_FoldParams);
        m_ModelParams = m_FoldParams;
        m_ShapesDirty = true;
        m_AnalysisValid = false;
        m_Figure.Frame(m_InputBounds);
        m_Ball.Frame(m_InputBounds);
        m_Fold = 1.0f;
        m_Status = m_Shape->ok ? "Loaded " + name + ". " + m_Shape->message + "." : "Loaded " + name + ", but: " + m_Shape->message;
    }

    void App::HandleFileDrops() {
        if (!IsFileDropped()) return;
        FilePathList files = LoadDroppedFiles();
        int toLoad = -1, skipped = 0;
        std::string design;
        for (unsigned i = 0; i < files.count; i++) {
            const char* path = files.paths[i];
            if (IsFileExtension(path, ".txt")) { design = path; continue; }
            if (!Utils::IsModelFile(path)) { skipped++; continue; }
            int idx = AddToLibrary({ GetFileName(path), path });
            if (toLoad < 0) toLoad = idx;
        }
        UnloadDroppedFiles(files);
        if (toLoad >= 0) LoadAsset(toLoad);
        if (!design.empty()) {
            snprintf(m_DesignPath, sizeof(m_DesignPath), "%s", design.c_str());
            LoadDesign();
        }
        if (skipped) m_Status += " (" + std::to_string(skipped) + " unsupported file(s) ignored)";
    }

    void App::AcceptAssetDrop(const Viewport& vp) {
        Rectangle r = vp.ScreenRectangle();
        ImRect rect(r.x, r.y, r.x + r.width, r.y + r.height);
        if (!ImGui::BeginDragDropTargetCustom(rect, ImGui::GetID("##viewport_drop"))) return;
        if (const ImGuiPayload* p = ImGui::AcceptDragDropPayload("ASSET")) LoadAsset(*(const int*)p->Data);
        ImGui::EndDragDropTarget();
    }

    // ---- jobs ------------------------------------------------------------------------------------------------------

    bool App::Busy() const {
        return m_DecomposeJob.Busy() || m_FoldJob.Busy() || m_RestoreJob.Busy();
    }

    const Design& App::Shown() const {
        return (Busy() && m_Preview) ? *m_Preview : m_Design;
    }

    void App::PollJobs() {
        m_Preview.reset();
        const JobControl* c = m_DecomposeJob.Busy() ? m_DecomposeJob.Control() : m_FoldJob.Busy() ? m_FoldJob.Control() : nullptr;
        if (c) {
            std::shared_ptr<const Design> p = c->Preview(nullptr);
            if (p) {
                // previews of a decomposition carry no sphere; keep ours
                Design d = *p;
                if (m_DecomposeJob.Busy()) d.ball = m_Design.ball;
                m_Preview = std::make_shared<const Design>(std::move(d));
            }
        }

        DecomposeResult dr;
        if (m_DecomposeJob.Poll(&dr)) {
            m_Pipeline = Pipeline::Idle;
            m_Status = dr.message;
            if (dr.ok) {
                if (m_DecomposeParams.refineOnly && dr.shapes.size() == m_Design.parts.size()) {
                    // shape refinement keeps the order, names and hinges
                    for (size_t i = 0; i < dr.shapes.size(); i++) m_Design.parts[i].shape = dr.shapes[i];
                } else {
                    Design d;
                    d.ball = m_Design.ball;
                    for (const SuperFrustum& s : dr.shapes) d.AddPart(s, -1);
                    m_Design = d;
                    m_Selected = -1;
                    BuildTree();
                }
                if (m_RunAll && !dr.cancelled) StartFold(false);
                else m_RunAll = false;
            } else {
                m_RunAll = false;
            }
        }
        FoldResult fr;
        if (m_FoldJob.Poll(&fr)) {
            m_Pipeline = Pipeline::Idle;
            m_Status = fr.message;
            if (m_FoldJob.Control()) m_LastHistory = m_FoldJob.Control()->History();
            if (fr.ok) {
                int keepId = m_Selected >= 0 && m_Selected < (int)m_Design.parts.size() ? m_Design.parts[m_Selected].id : -1;
                m_Design = fr.design;
                m_Selected = keepId >= 0 ? m_Design.IndexOf(keepId) : -1;
                m_Fold = 1.0f;
                if (m_RunAll && !fr.cancelled) StartRestore();
                else m_RunAll = false;
            } else {
                m_RunAll = false;
            }
        }
        RestoreResult rr;
        if (m_RestoreJob.Poll(&rr)) {
            m_Pipeline = Pipeline::Idle;
            m_RunAll = false;
            m_Status = rr.message;
            if (rr.ok) {
                ClearRestored();
                for (RestoredPart& p : rr.parts) {
                    RestoredModel m;
                    m.partId = p.partId;
                    m.model = Utils::TriMeshToModel(p.mesh, m_Shader);
                    m.mesh = std::move(p.mesh);
                    m_Restored.push_back(std::move(m));
                }
                m_RestoredFor = rr.design;
                m_ShowPieces = true;
            }
        }
    }

    void App::StartDecompose(bool refineOnly, int onlyIndex) {
        if (Busy() || !m_Mesh) return;
        m_DecomposeParams.refineOnly = refineOnly;
        m_DecomposeParams.onlyIndex = onlyIndex;
        DecomposeParams params = m_DecomposeParams;
        std::vector<SuperFrustum> initial;
        if (refineOnly)
            for (const Part& p : m_Design.parts) initial.push_back(p.shape);
        std::shared_ptr<const TriMesh> mesh = m_Mesh;
        m_DecomposeJob.Start([mesh, params, initial](JobControl& c) { return Decompose(mesh, params, initial, c); });
        m_Pipeline = Pipeline::Decompose;
        m_Status = refineOnly ? "Fitting superfrustums to the mesh..." : "Decomposing the mesh into superfrustums...";
    }

    void App::StartFold(bool keepTree) {
        if (Busy() || !m_Shape || !m_Shape->ok) return;
        if (m_Design.parts.empty()) { m_Status = "There are no superfrustums to fold yet."; m_RunAll = false; return; }
        FoldParams params = m_FoldParams;
        params.keepTree = keepTree;
        Design start = m_Design;
        std::shared_ptr<const ShapeAnalysis> shape = m_Shape;
        m_FoldJob.Start([start, shape, params](JobControl& c) { return OptimizeFold(start, shape, params, c); });
        m_Pipeline = Pipeline::Fold;
        m_Fold = 1.0f;
        m_Status = "Optimising the fold (stochastic rewrite descent)...";
    }

    void App::StartRestore() {
        if (Busy() || !m_Mesh) return;
        if (m_Design.parts.empty()) { m_Status = "There are no parts to build."; m_RunAll = false; return; }
        RestoreParams params = m_RestoreParams;
        Design design = m_Design;
        std::shared_ptr<const TriMesh> mesh = m_Mesh;
        m_RestoreJob.Start([design, mesh, params](JobControl& c) { return RestoreDetail(design, mesh, params, c); });
        m_Pipeline = Pipeline::Restore;
        m_Status = "Restoring detail and building the pieces...";
    }

    void App::RunAll() {
        if (Busy()) return;
        m_RunAll = true;
        StartDecompose(false, -1);
    }

    void App::BuildTree() {
        if (!m_FoldModel.Valid() || m_Design.parts.empty()) return;
        FoldState s = m_FoldModel.MakeState(m_Design);
        BuildFoldTree(s, m_FoldModel);
        int keepId = m_Selected >= 0 && m_Selected < (int)m_Design.parts.size() ? m_Design.parts[m_Selected].id : -1;
        m_Design = s.design;
        m_Selected = keepId >= 0 ? m_Design.IndexOf(keepId) : -1;
        m_Fold = 1.0f;
    }

    void App::AimSelected() {
        if (!m_FoldModel.Valid() || m_Selected < 0 || m_Selected >= (int)m_Design.parts.size()) return;
        FoldState s = m_FoldModel.MakeState(m_Design);
        AimHinge(s, m_FoldModel, m_Selected);
        m_Design = s.design;
        m_Fold = 1.0f;
    }

    void App::SnapPivotToContact(int part) {
        if (!m_FoldModel.Valid() || part < 0 || part >= (int)m_Design.parts.size()) return;
        int parent = m_Design.parts[part].hinge.parent;
        if (parent < 0) return;
        FoldState s = m_FoldModel.MakeState(m_Design);
        Vector3 p;
        int count = 0;
        if (m_FoldModel.ContactPoint(s, part, parent, &p, &count)) s.design.parts[part].hinge.pivot = p;
        else m_FoldModel.ProjectPivot(s, part);
        m_Design.parts[part].hinge.pivot = s.design.parts[part].hinge.pivot;
    }

    void App::UpdateAnalysis() {
        if (!m_Shape || !m_Shape->ok) { m_AnalysisValid = false; return; }
        if (m_ModelParams != m_FoldParams) {
            m_ModelParams = m_FoldParams;
            m_FoldModel.SetParams(m_FoldParams);
            m_ShapesDirty = true;
        }
        const Design& d = Shown();
        bool shapes = m_ShapesDirty || m_Analysis.design.parts.size() != d.parts.size();
        for (size_t i = 0; !shapes && i < d.parts.size(); i++)
            shapes = d.parts[i].id != m_Analysis.design.parts[i].id || d.parts[i].shape != m_Analysis.design.parts[i].shape;
        if (!shapes && m_AnalysisValid && m_Analysis.design == d) return;
        if (shapes) {
            m_Analysis = m_FoldModel.MakeState(d);
            m_ShapesDirty = false;
        } else {
            m_Analysis.design = d;
        }
        m_Visual = FoldVisual{};
        m_FoldModel.Evaluate(m_Analysis, &m_Metrics, &m_Visual);
        m_AnalysisValid = !d.parts.empty();
    }

    void App::ClearRestored() {
        for (RestoredModel& m : m_Restored) UnloadModel(m.model);
        m_Restored.clear();
        m_RestoredFor = Design{};
    }

    bool App::RestoredCurrent() const {
        if (m_Restored.empty()) return false;
        const Design& d = m_Design;
        if (d.parts.size() != m_RestoredFor.parts.size() || !(d.ball == m_RestoredFor.ball)) return false;
        for (size_t i = 0; i < d.parts.size(); i++)
            if (d.parts[i].id != m_RestoredFor.parts[i].id || d.parts[i].shape != m_RestoredFor.parts[i].shape ||
                !(d.parts[i].hinge == m_RestoredFor.parts[i].hinge))
                return false;
        return true;
    }

    // ---- files -----------------------------------------------------------------------------------------------------

    void App::ExportParts() {
        std::string base = m_ExportPath;
        std::vector<const TriMesh*> meshes;
        std::vector<std::string> names;
        std::vector<Rigid> open, closed;
        const Design& d = m_Restored.empty() ? m_Design : m_RestoredFor;
        std::vector<Rigid> xf = d.Transforms(1.0f);
        std::vector<TriMesh> coarse;
        if (!m_Restored.empty()) {
            for (const RestoredModel& m : m_Restored) {
                int i = d.IndexOf(m.partId);
                if (i < 0) continue;
                meshes.push_back(&m.mesh);
                names.push_back(d.parts[i].name);
                open.push_back(Rigid{});
                closed.push_back(xf[i]);
            }
        } else {
            // no pieces built yet: export the superfrustums themselves
            coarse.reserve(d.parts.size());
            for (const Part& p : d.parts) {
                const SuperFrustum& sf = p.shape;
                coarse.push_back(Utils::MeshImplicit([&](Vector3 q) { return sf.LocalDistance(q); }, sf.LocalBounds(), 40));
            }
            for (size_t i = 0; i < d.parts.size(); i++) {
                meshes.push_back(&coarse[i]);
                names.push_back(d.parts[i].name);
                open.push_back(d.parts[i].shape.Pose());
                closed.push_back(d.parts[i].shape.Pose().Then(xf[i]));
            }
        }
        if (meshes.empty()) { m_Status = "Nothing to export."; return; }
        std::string err;
        if (!Utils::WriteOBJ(base + "_open.obj", meshes, open, names, &err) ||
            !Utils::WriteOBJ(base + "_closed.obj", meshes, closed, names, &err)) {
            m_Status = err;
            return;
        }
        FILE* fp = fopen((base + "_hinges.txt").c_str(), "w");
        if (fp) {
            fprintf(fp, "Bakugan Everything - hinge sheet\n");
            fprintf(fp, "sphere centre %.5f %.5f %.5f  radius %.5f\n\n", d.ball.center.x, d.ball.center.y, d.ball.center.z, d.ball.radius);
            fprintf(fp, "%-14s %-14s %-8s %-30s %-30s %s\n", "part", "parent", "type", "pivot (open pose)", "axis (open pose)", "fold");
            for (const Part& p : d.parts) {
                const Hinge& h = p.hinge;
                const char* parent = h.parent >= 0 ? d.parts[h.parent].name.c_str() : "-";
                const char* type = h.parent < 0 ? "root" : h.locked ? "rigid" : "spring";
                fprintf(fp, "%-14s %-14s %-8s %9.4f %9.4f %9.4f   %8.4f %8.4f %8.4f   %6.1f deg\n", p.name.c_str(), parent, type,
                        h.pivot.x, h.pivot.y, h.pivot.z, h.axis.x, h.axis.y, h.axis.z, d.FoldAngle(&p - d.parts.data()) * RAD2DEG);
            }
            fprintf(fp, "\nSpring hinges are uni-directional: they rotate from 0 (open, the figure) up to the fold angle\n"
                        "(closed, the ball) about +axis by the right-hand rule, and their springs push them back open.\n");
            fclose(fp);
        }
        m_Status = "Exported " + std::to_string(meshes.size()) + (m_Restored.empty() ? " superfrustum" : " restored") +
                   " parts to " + base + "_open.obj, " + base + "_closed.obj and " + base + "_hinges.txt";
    }

    void App::SaveDesign() {
        std::string text = m_Design.Serialize();
        if (SaveFileText(m_DesignPath, (char*)text.c_str())) m_Status = std::string("Saved the design to ") + m_DesignPath;
        else m_Status = std::string("Could not write ") + m_DesignPath;
    }

    void App::LoadDesign() {
        if (Busy()) return;
        char* text = LoadFileText(m_DesignPath);
        if (!text) { m_Status = std::string("Could not read ") + m_DesignPath; return; }
        Design d;
        std::string err;
        bool ok = d.Deserialize(text, &err);
        UnloadFileText(text);
        if (!ok) { m_Status = err; return; }
        m_Design = d;
        m_Selected = -1;
        m_Status = std::string("Loaded the design from ") + m_DesignPath + " (" + std::to_string(d.parts.size()) + " parts)";
    }

    // ---- editing helpers -------------------------------------------------------------------------------------------

    void App::Select(int part) {
        m_Selected = part >= 0 && part < (int)m_Design.parts.size() ? part : -1;
    }

    int App::AddPart(const SuperFrustum& shape) {
        int i = m_Design.AddPart(shape, -1);
        if (m_Design.Root() != i) {
            // attach new parts to the selection (or the root) so they fold with it
            int parent = m_Selected >= 0 ? m_Selected : m_Design.Root();
            m_Design.parts[i].hinge.parent = parent;
            m_Design.parts[i].hinge.pivot = shape.position;
            m_Design.parts[i].hinge.locked = true;
        }
        return i;
    }

    void App::DeleteSelected() {
        if (m_Selected < 0 || m_Selected >= (int)m_Design.parts.size()) return;
        int parent = m_Design.parts[m_Selected].hinge.parent;
        m_Design.RemovePart(m_Selected);
        Select(parent > m_Selected ? parent - 1 : parent);
    }

    void App::DuplicateSelected() {
        if (m_Selected < 0 || m_Selected >= (int)m_Design.parts.size()) return;
        Part src = m_Design.parts[m_Selected];
        SuperFrustum sf = src.shape;
        float shift = 0.6f * fmaxf(sf.size.x, 0.1f);
        sf.position = Vector3Add(sf.position, { shift, 0, 0 });
        int i = m_Design.AddPart(sf, -1, src.name + "_copy");
        Hinge h = src.hinge;
        h.pivot = Vector3Add(h.pivot, { shift, 0, 0 });
        m_Design.parts[i].hinge = h;
        if (h.parent < 0 && m_Design.Root() != i) m_Design.parts[i].hinge.parent = m_Design.Root();
        Select(i);
    }

    // ---- frame -----------------------------------------------------------------------------------------------------

    void App::UpdateFoldAnimation() {
        float dt = fminf(GetFrameTime(), 1.0f / 30.0f);
        if (m_FoldAnim == 1) {
            // springs push every hinge open; the hinge stops at 0 and bounces a little
            m_FoldVelocity += (-60.0f * m_Fold - 4.0f * m_FoldVelocity) * dt;
            m_Fold += m_FoldVelocity * dt;
            if (m_Fold < 0.0f) { m_Fold = 0.0f; m_FoldVelocity = -m_FoldVelocity * 0.35f; }
            if (m_Fold < 0.002f && fabsf(m_FoldVelocity) < 0.05f) { m_Fold = 0.0f; m_FoldAnim = 0; }
        } else if (m_FoldAnim == 2) {
            m_Fold = fminf(1.0f, m_Fold + dt * 0.7f);
            if (m_Fold >= 1.0f) m_FoldAnim = 0;
        }
        if (m_Loop && m_FoldAnim == 0) {
            m_LoopTimer += dt;
            if (m_LoopTimer > 0.8f) {
                m_LoopTimer = 0.0f;
                m_FoldAnim = m_Fold > 0.5f ? 1 : 2;
                m_FoldVelocity = 0.0f;
            }
        }
    }

    void App::Frame() {
        HandleFileDrops();
        PollJobs();
        UpdateAnalysis();
        UpdateFoldAnimation();
        PurgePartMeshes(Shown());

        float W = (float)GetScreenWidth(), H = (float)GetScreenHeight();
        float sidebar = fminf(390.0f, W * 0.3f);
        float half = (W - sidebar) * 0.5f;

        rlImGuiBegin();
        ImGuizmo::BeginFrame();
        DrawSidebar(sidebar, H);
        DrawViewportWindow(m_Figure, ImVec2(sidebar, 0), ImVec2(half, H), true);
        DrawViewportWindow(m_Ball, ImVec2(sidebar + half, 0), ImVec2(W - sidebar - half, H), false);
        HandleKeys();
        TrackHistory();

        RenderFigure();
        RenderBall();

        BeginDrawing();
        ClearBackground({ 20, 20, 24, 255 });
        rlImGuiEnd();
        EndDrawing();
    }
}
