#include "app.hpp"
#include "util/utils.hpp"
#include <imgui_internal.h>
#include <rlImGui.h>
#include <nfd.h>
#include <ImGuizmo.h>
#include <algorithm>
#include <raymath.h>
#include <raylib.h>
#include <imgui.h>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>

#define MAX_UNDO 200

namespace Bakugan {
    App::App() {
        SetTraceLogLevel(LOG_NONE);
        SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_WINDOW_HIGHDPI | FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
        InitWindow(1280, 720, "Bakugan Hinges");
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
        if (!m_Sdf.Init()) m_Exact = false;   // no float precision / GLSL 3.3 trouble: draw meshes instead
        LoadTemplate(0);
        m_UndoStack.clear();
        m_Committed = Capture();
    }

    App::~App() {
        m_BuildJob.Stop();
        m_SweepJob.Stop();
        m_ExportJob.Stop();
        ClearParts();
        for (auto& [id, sm] : m_ShapeModels)
            if (sm.loaded) UnloadModel(sm.model);
        if (m_HasReference) UnloadModel(m_Reference);
        m_Sdf.Unload();
        m_DesignView.Unload();
        m_OutputView.Unload();
        UnloadShader(m_Shader);
        rlImGuiShutdown();
        CloseWindow();
    }

    void App::Run() {
        while (!WindowShouldClose()) Frame();
    }

    // ---- history ---------------------------------------------------------------------------------------------------

    EditState App::Capture() const {
        return { m_Design, m_Sel };
    }

    void App::Restore(const EditState& st) {
        m_Design = st.design;
        m_Sel = st.selection;
        SanitizeSelection();
    }

    bool App::Interacting() {
        return ImGui::IsAnyItemActive() || ImGuizmo::IsUsingAny() || ImGui::IsMouseDown(ImGuiMouseButton_Left);
    }

    void App::TrackHistory() {
        if (Interacting()) return;
        if (m_Committed.design == m_Design) {
            m_Committed.selection = m_Sel;
            return;
        }
        m_UndoStack.push_back(std::move(m_Committed));
        if (m_UndoStack.size() > MAX_UNDO) m_UndoStack.erase(m_UndoStack.begin());
        m_RedoStack.clear();
        m_Committed = Capture();
    }

    void App::Undo() {
        TrackHistory();
        if (m_UndoStack.empty() || Interacting()) return;
        m_RedoStack.push_back(Capture());
        m_Committed = std::move(m_UndoStack.back());
        m_UndoStack.pop_back();
        Restore(m_Committed);
    }

    void App::Redo() {
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
            if (ImGui::IsKeyPressed(ImGuiKey_G)) { if (io.KeyShift) UngroupSelection(); else GroupSelection(); }
            if (ImGui::IsKeyPressed(ImGuiKey_D)) DuplicateSelection();
            return;
        }
        bool overView = m_DesignView.Hovered() || m_OutputView.Hovered();
        if (!overView) return;
        if (ImGui::IsKeyPressed(ImGuiKey_F)) {
            if (m_DesignView.Hovered()) FrameView(m_DesignView);
            if (m_OutputView.Hovered()) FrameView(m_OutputView);
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Space)) {
            m_Anim = m_T > 0.5f ? -1 : 1;
            m_LoopTimer = 0.0f;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_W)) m_GizmoOp = GizmoOp::Translate;
        if (ImGui::IsKeyPressed(ImGuiKey_E)) m_GizmoOp = GizmoOp::Rotate;
        if (ImGui::IsKeyPressed(ImGuiKey_R)) m_GizmoOp = GizmoOp::Scale;
        if (ImGui::IsKeyPressed(ImGuiKey_Q))
            m_PickMode = m_PickMode == PickMode::Shape ? PickMode::Feature : m_PickMode == PickMode::Feature ? PickMode::Body : PickMode::Shape;
        if (ImGui::IsKeyPressed(ImGuiKey_X) && !m_Sel.shapes.empty()) ToggleCut();
        if (ImGui::IsKeyPressed(ImGuiKey_H)) {
            if (io.KeyAlt) m_Hidden.clear();
            else HideSelection();
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) ClearSelection();
        if (ImGui::IsKeyPressed(ImGuiKey_Delete) || ImGui::IsKeyPressed(ImGuiKey_Backspace)) DeleteSelection();
        if (ImGui::IsKeyPressed(ImGuiKey_D) && io.KeyShift) DuplicateSelection();
        if (ImGui::IsKeyPressed(ImGuiKey_A) && io.KeyShift) AddPreset(Preset::Box);
    }

    // ---- files -----------------------------------------------------------------------------------------------------

    void App::SetDesign(const Design& d, const std::string& status) {
        // the history keeps the previous design, so this can be undone
        m_Design = d;
        m_Sel = Selection{};
        m_T = 0.0f;
        m_Anim = 0;
        m_HasSweep = false;
        FrameView(m_DesignView);
        FrameView(m_OutputView);
        m_Status = status;
    }

    void App::LoadTemplate(int index) {
        std::vector<std::string> names = TemplateNames();
        if (index < 0 || index >= (int)names.size()) return;
        m_TemplateIndex = index;
        SetDesign(MakeTemplate(index), "Loaded the " + names[index] + " template. Space plays the motion.");
    }

    bool App::OpenFile(const std::string& path) {
        if (path.empty()) return false;
        if (!FileExists(path.c_str())) { m_Status = "File not found: " + path; return false; }
        if (IsFileExtension(path.c_str(), ".txt")) {
            snprintf(m_DesignPath, sizeof(m_DesignPath), "%s", path.c_str());
            LoadDesign();
            return true;
        }
        if (Utils::IsModelFile(path.c_str())) {
            LoadReference(path);
            return true;
        }
        m_Status = "Unsupported file type: " + path;
        return false;
    }

    void App::HandleFileDrops() {
        if (!IsFileDropped()) return;
        FilePathList files = LoadDroppedFiles();
        std::vector<std::string> paths;
        for (unsigned i = 0; i < files.count; i++) paths.push_back(files.paths[i]);
        UnloadDroppedFiles(files);
        for (const std::string& p : paths) OpenFile(p);
    }

    // Native file dialogs (nativefiledialog: GTK3 on Linux, the system dialogs on Windows). They block until closed,
    // and work where drag and drop doesn't, e.g. Wayland compositors talking to this XWayland window.
    // filters: "txt" or "obj,gltf,glb". start: a file or folder to open the dialog in. False when cancelled or failed.
    bool App::FileDialog(bool save, const char* filters, const std::string& start, std::string* out) {
        // nfd wants an existing absolute folder to start in
        std::string folder;
        if (!start.empty()) {
            std::string abs = start[0] == '/' || (start.size() > 1 && start[1] == ':') ? start
                                                                                        : std::string(GetWorkingDirectory()) + "/" + start;
            folder = DirectoryExists(abs.c_str()) ? abs : GetDirectoryPath(abs.c_str());
            if (!DirectoryExists(folder.c_str())) folder.clear();
        }
        if (folder.empty()) folder = GetWorkingDirectory();
        nfdchar_t* path = nullptr;
        nfdresult_t r = save ? NFD_SaveDialog(filters, folder.c_str(), &path) : NFD_OpenDialog(filters, folder.c_str(), &path);
        // take focus back from the dialog: rlImGui ignores the mouse while the window is unfocused, and not every
        // window manager hands focus back by itself
        SetWindowFocused();
        if (r == NFD_OKAY && path) {
            *out = path;
            free(path);
            return true;
        }
        if (r == NFD_ERROR) {
            const char* e = NFD_GetError();
            m_Status = std::string("The file dialog failed") + (e ? std::string(": ") + e : "") + ". You can type a path instead.";
        } else {
            m_Status = "No file chosen.";
        }
        return false;
    }

    static bool EndsWith(const std::string& s, const char* suffix) {
        size_t n = strlen(suffix);
        if (s.size() < n) return false;
        for (size_t i = 0; i < n; i++)
            if (tolower((unsigned char)s[s.size() - n + i]) != tolower((unsigned char)suffix[i])) return false;
        return true;
    }

    void App::BrowseDesign(bool save) {
        std::string path;
        if (!FileDialog(save, "txt", m_DesignPath, &path)) return;
        if (save && !EndsWith(path, ".txt")) path += ".txt";
        snprintf(m_DesignPath, sizeof(m_DesignPath), "%s", path.c_str());
        if (save) SaveDesign();
        else LoadDesign();
    }

    void App::BrowseExport() {
        std::string p;
        if (!FileDialog(true, "stl", m_ExportPath, &p)) return;
        if (EndsWith(p, ".stl")) p.resize(p.size() - 4);   // a prefix: <prefix>_<part>.stl
        snprintf(m_ExportPath, sizeof(m_ExportPath), "%s", p.c_str());
        StartExport();
    }

    void App::BrowseReference() {
        std::string path;
        if (!FileDialog(false, "obj,gltf,glb,iqm,vox,m3d", m_ReferencePath, &path)) return;
        snprintf(m_ReferencePath, sizeof(m_ReferencePath), "%s", path.c_str());
        LoadReference(path);
    }

    void App::LoadReference(const std::string& path) {
        Model m;
        std::string err;
        if (!Utils::LoadModelFile(path, &m, &err)) { m_Status = err; return; }
        if (m_HasReference) UnloadModel(m_Reference);
        m_Reference = m;
        m_HasReference = true;
        m_ShowReference = true;
        m_ReferenceName = GetFileName(path.c_str());
        for (int i = 0; i < m_Reference.materialCount; i++) m_Reference.materials[i].shader = m_Shader;
        // millimetre models are used as they are; anything tiny or huge is fitted to 80 mm, standing on the ground
        BoundingBox b = GetModelBoundingBox(m_Reference);
        m_ReferenceBox = b;
        m_ReferenceRotation = { 0, 0, 0 };
        m_ReferenceOpen = true;
        Vector3 size = Vector3Subtract(b.max, b.min);
        float extent = fmaxf(size.x, fmaxf(size.y, size.z));
        m_ReferenceScale = (extent > 5.0f && extent < 1000.0f) || extent < 1e-6f ? 1.0f : 80.0f / extent;
        m_ReferenceOffset = { -(b.min.x + b.max.x) * 0.5f * m_ReferenceScale, -b.min.y * m_ReferenceScale,
                              -(b.min.z + b.max.z) * 0.5f * m_ReferenceScale };
        m_Status = "Reference mesh " + m_ReferenceName + " loaded (" + std::to_string((int)(extent * m_ReferenceScale)) + " mm), " +
                   "shown solid: superfrustums show where they poke out of it. It is never exported.";
        FrameView(m_DesignView);
    }

    void App::SaveDesign() {
        std::string text = m_Design.Serialize();
        std::string err;
        if (Utils::WriteTextFile(m_DesignPath, text, &err)) m_Status = std::string("Saved the design to ") + m_DesignPath;
        else m_Status = err;
    }

    void App::LoadDesign() {
        char* text = LoadFileText(m_DesignPath);
        if (!text) { m_Status = std::string("Could not read ") + m_DesignPath; return; }
        Design d;
        std::string err;
        bool ok = d.Deserialize(text, &err);
        UnloadFileText(text);
        if (!ok) { m_Status = err; return; }
        SetDesign(d, std::string("Loaded ") + m_DesignPath + " (" + std::to_string(d.shapes.size()) + " shapes, " +
                     std::to_string(d.Bodies().size()) + " bodies, " + std::to_string(d.hinges.size()) + " hinges)");
    }

    // ---- jobs ------------------------------------------------------------------------------------------------------

    void App::ClearParts() {
        for (PartModel& p : m_PartModels) UnloadModel(p.model);
        m_PartModels.clear();
        m_Parts.reset();
        m_HasParts = false;
    }

    bool App::PartsCurrent() const {
        return m_HasParts && SameGeometry(m_PartsFor, m_Design);
    }

    bool App::SweepCurrent() const {
        return m_HasSweep && PartsCurrent() && m_Sweep.design == m_Design && m_Sweep.params == m_SweepParams;
    }

    const SweepStep* App::CurrentStep() const {
        if (!m_HasSweep || !m_ShowContacts) return nullptr;
        int k = m_Sweep.StepAt(m_T);
        return k >= 0 ? &m_Sweep.steps[k] : nullptr;
    }

    void App::PollJobs() {
        BuildResult br;
        if (m_BuildJob.Poll(&br)) {
            if (br.ok) {
                // keep the GPU models of parts that didn't change, upload the rest
                std::vector<PartModel> models;
                for (const BuiltPart& p : br.parts) {
                    PartModel pm;
                    auto it = std::find_if(m_PartModels.begin(), m_PartModels.end(), [&](const PartModel& m) { return !m.key.empty() && m.key == p.key; });
                    if (it != m_PartModels.end()) {
                        pm = *it;
                        it->key.clear();
                        it->model = Model{};
                    } else {
                        pm.model = Utils::TriMeshToModel(p.mesh, m_Shader);
                        pm.triangles = p.mesh.indices.size() / 3;
                        pm.key = p.key;
                    }
                    pm.body = p.body;
                    pm.pinHinge = p.pinHinge;
                    models.push_back(pm);
                }
                for (PartModel& old : m_PartModels)
                    if (old.model.meshCount > 0) UnloadModel(old.model);
                m_PartModels = std::move(models);
                m_Parts = std::make_shared<const std::vector<BuiltPart>>(std::move(br.parts));
                m_PartsFor = br.design;
                m_PartsVoxel = br.voxel;
                m_HasParts = true;
                m_SpareParts.clear();
                char buf[96];
                snprintf(buf, sizeof(buf), "last update %.0f ms, %d of %d parts remeshed", (GetTime() - m_BuildStart) * 1000.0,
                         br.rebuilt, (int)m_Parts->size());
                m_BuildStats = buf;
            } else if (br.cancelled) {
                for (BuiltPart& p : br.parts) {
                    bool known = std::any_of(m_SpareParts.begin(), m_SpareParts.end(), [&](const BuiltPart& q) { return q.key == p.key; });
                    if (!known) m_SpareParts.push_back(std::move(p));
                }
                if (m_SpareParts.size() > 32) m_SpareParts.erase(m_SpareParts.begin(), m_SpareParts.end() - 32);
            }
        }
        SweepResult sr;
        if (m_SweepJob.Poll(&sr) && sr.ok) {
            char buf[96];
            snprintf(buf, sizeof(buf), "last check %.0f ms, %d of %d body pairs rechecked", (GetTime() - m_SweepStart) * 1000.0,
                     sr.pairsChecked, sr.pairsChecked + sr.pairsReused);
            m_SweepStats = buf;
            m_Sweep = std::move(sr);
            m_HasSweep = true;
        }
        ExportResult er;
        if (m_ExportJob.Poll(&er)) {
            m_Status = er.message;
            if (er.ok) m_ExportedFiles = er.files;
        }
    }

    // Keeps the parts and the collision sweep in step with the design, once the user lets go of whatever they drag.
    // Only what changed is redone: parts whose geometry is the same are reused, and so are the collision checks of
    // pairs of bodies whose parts and relative motion are the same. A job that went stale is stopped without waiting
    // for it (the UI never blocks on a worker); the next one starts once it has wound down.
    void App::UpdateJobs() {
        if (Interacting()) return;
        float voxel = PreviewVoxel(m_Design, m_PreviewResolution);
        bool partsStale = !m_HasParts || !SameGeometry(m_PartsFor, m_Design) || m_PartsVoxel != voxel;
        if (partsStale) {
            if (m_BuildJob.Busy()) {
                if (!SameGeometry(m_BuildingFor, m_Design) || m_BuildingVoxel != voxel) m_BuildJob.Cancel();
                return;
            }
            if (m_SweepJob.Busy()) m_SweepJob.Cancel();   // its parts are about to change
            Design d = m_Design;
            m_BuildingFor = d;
            m_BuildingVoxel = voxel;
            // the worker reads the current parts through shared pointers: nothing is copied on the UI thread
            std::shared_ptr<const std::vector<BuiltPart>> current = m_Parts;
            auto spare = std::make_shared<const std::vector<BuiltPart>>(std::move(m_SpareParts));
            m_SpareParts.clear();
            m_BuildStart = GetTime();
            m_BuildJob.Start([d, voxel, current, spare](JobControl& c) {
                std::vector<const BuiltPart*> reuse;
                for (const BuiltPart& p : *spare) reuse.push_back(&p);
                if (current)
                    for (const BuiltPart& p : *current) reuse.push_back(&p);
                return BuildParts(d, voxel, 0.0f, 5000, c, nullptr, 0.0f, &reuse);
            });
            return;
        }
        m_SweepParams.ease = m_Ease;
        bool sweepStale = !m_HasSweep || !(m_Sweep.design == m_Design) || m_Sweep.params != m_SweepParams;
        if (!sweepStale) return;
        if (m_SweepJob.Busy()) {
            if (!(m_SweepingFor == m_Design) || m_SweepingParams != m_SweepParams) m_SweepJob.Cancel();
            return;
        }
        Design d = m_Design;
        SweepParams p = m_SweepParams;
        std::shared_ptr<const std::vector<BuiltPart>> parts = m_Parts;
        std::shared_ptr<SweepCache> cache = m_SweepCache;
        m_SweepingFor = d;
        m_SweepingParams = p;
        m_SweepStart = GetTime();
        m_SweepJob.Start([d, parts, p, cache](JobControl& c) { return SweepCollisions(d, parts, p, c, cache); });
    }

    void App::StartExport() {
        if (m_ExportJob.Busy()) return;
        if (m_Design.Bodies().empty()) { m_Status = "Nothing to export: add some shapes first."; return; }
        ExportParams p = m_ExportParams;
        p.prefix = m_ExportPath;
        p.ease = m_Ease;
        Design d = m_Design;
        m_ExportedFiles.clear();
        m_ExportJob.Start([d, p](JobControl& c) { return ExportSTL(d, p, c); });
        m_Status = "Exporting STL parts...";
    }

    // ---- selection -------------------------------------------------------------------------------------------------

    void App::ClearSelection() {
        m_Sel = Selection{};
    }

    void App::SelectShape(int id, bool additive) {
        if (!additive) {
            m_Sel = Selection{};
            if (id >= 0) m_Sel.shapes.push_back(id);
            return;
        }
        m_Sel.group = -1;
        m_Sel.hinge = -1;
        auto it = std::find(m_Sel.shapes.begin(), m_Sel.shapes.end(), id);
        if (it != m_Sel.shapes.end()) m_Sel.shapes.erase(it);
        else if (id >= 0) m_Sel.shapes.push_back(id);
    }

    void App::SelectGroup(int id) {
        m_Sel = Selection{};
        m_Sel.group = id;
    }

    void App::SelectHinge(int id) {
        m_Sel = Selection{};
        m_Sel.hinge = id;
    }

    void App::SanitizeSelection() {
        std::vector<int> keep;
        for (int id : m_Sel.shapes)
            if (m_Design.ShapeIndex(id) >= 0) keep.push_back(id);
        m_Sel.shapes = keep;
        if (m_Design.GroupIndex(m_Sel.group) < 0) m_Sel.group = -1;
        if (m_Design.HingeIndex(m_Sel.hinge) < 0) m_Sel.hinge = -1;
    }

    int App::SelectedShape() const {
        return m_Sel.shapes.size() == 1 ? m_Design.ShapeIndex(m_Sel.shapes[0]) : -1;
    }

    int App::SelectedGroup() const {
        return m_Design.GroupIndex(m_Sel.group);
    }

    int App::SelectedHinge() const {
        return m_Design.HingeIndex(m_Sel.hinge);
    }

    std::vector<int> App::SelectedShapeIndices() const {
        if (m_Sel.group >= 0) return m_Design.GroupShapes(m_Sel.group);
        std::vector<int> out;
        for (int id : m_Sel.shapes) {
            int s = m_Design.ShapeIndex(id);
            if (s >= 0) out.push_back(s);
        }
        return out;
    }

    int App::ContextGroup() const {
        if (SelectedGroup() >= 0) return m_Sel.group;
        if (!m_Sel.shapes.empty()) {
            int s = m_Design.ShapeIndex(m_Sel.shapes[0]);
            if (s >= 0) return m_Design.shapes[s].group;
        }
        return -1;
    }

    // ---- editing ---------------------------------------------------------------------------------------------------

    void App::AddPreset(Preset p) {
        SuperFrustum sf;
        sf.size = { 20, 20, 20 };
        Quaternion up = Utils::EulerDegreesToQuaternion({ -90, 0, 0 });
        const char* name = "Box";
        switch (p) {
            case Preset::Box: break;
            case Preset::Cylinder: name = "Cylinder"; sf.rotation = up; sf.rounding = 1.0f; break;
            case Preset::Cone: name = "Cone"; sf.rotation = up; sf.rounding = 1.0f; sf.taper = 0.0f; break;
            case Preset::Ball: name = "Ball"; sf.size = { 1, 1, 1 }; sf.rounding = 1.0f; sf.dilation = 9.5f; break;
            case Preset::Wedge: name = "Wedge"; sf.rotation = up; sf.size = { 20, 10, 20 }; sf.taper = 0.0f; break;
            case Preset::Tube: name = "Tube"; sf.rotation = up; sf.rounding = 1.0f; sf.onion = 0.75f; break;
            case Preset::Arc: name = "Arc"; sf.size = { 6, 6, 40 }; sf.rounding = 1.0f; sf.bulge = 0.6f; break;
        }
        int group = ContextGroup();
        int sel = SelectedShape();
        if (sel >= 0) {
            sf.position = Vector3Add(m_Design.shapes[sel].sf.position, { 12, 0, 0 });
        } else if (group >= 0 && !m_Design.GroupShapes(group).empty()) {
            BoundingBox b = m_Design.BodyBounds(m_Design.BodyOf(group));
            sf.position = { (b.min.x + b.max.x) * 0.5f, b.max.y + 10.0f, (b.min.z + b.max.z) * 0.5f };
        } else if (!m_Design.shapes.empty()) {
            BoundingBox b = m_Design.Bounds();
            sf.position = { b.max.x + 15.0f, 10.0f, (b.min.z + b.max.z) * 0.5f };
        } else {
            sf.position = { 0, 10, 0 };
        }
        int s = m_Design.AddShape(sf, group, name);
        SelectShape(m_Design.shapes[s].id, false);
        m_Status = std::string("Added a ") + name + " to " + m_Design.GroupPath(m_Design.shapes[s].group) + ".";
    }

    void App::NewBody() {
        int g = m_Design.AddGroup("Body", -1);
        SelectGroup(m_Design.groups[g].id);
        m_Status = "New empty body " + m_Design.groups[g].name + ": shapes you add now go into it.";
    }

    void App::GroupSelection() {
        std::vector<int> groups;
        if (m_Sel.group >= 0) groups.push_back(m_Sel.group);
        if (m_Sel.shapes.empty() && groups.empty()) { m_Status = "Select shapes (Ctrl/Shift+click) to group."; return; }
        int id = m_Design.GroupItems(m_Sel.shapes, groups);
        if (id < 0) return;
        SelectGroup(id);
        int g = m_Design.GroupIndex(id);
        m_Status = g >= 0 && m_Design.groups[g].parent < 0
                     ? "Grouped into a new body (" + m_Design.groups[g].name + "): it moves and prints as one part."
                     : "Grouped into a feature (" + m_Design.groups[g].name + "): add it or cut it as one shape.";
    }

    void App::UngroupSelection() {
        int g = SelectedGroup();
        if (g < 0) { m_Status = "Select a group to ungroup."; return; }
        m_Design.Ungroup(g);
        ClearSelection();
        m_Status = "Ungrouped.";
    }

    void App::DeleteSelection() {
        if (SelectedHinge() >= 0) {
            m_Design.RemoveHinge(SelectedHinge());
        } else if (SelectedGroup() >= 0) {
            m_Design.RemoveGroup(SelectedGroup());
        } else {
            for (int id : m_Sel.shapes) m_Design.RemoveShape(m_Design.ShapeIndex(id));
        }
        ClearSelection();
    }

    void App::DuplicateSelection() {
        int g = SelectedGroup();
        if (g >= 0) {
            BoundingBox b = m_Design.BodyBounds(m_Sel.group);
            bool body = m_Design.groups[g].parent < 0;
            Vector3 off = body ? Vector3{ b.max.x - b.min.x + 10.0f, 0, 0 } : Vector3{ 10, 0, 0 };
            int n = m_Design.DuplicateGroup(g, m_Design.groups[g].parent, off);
            if (n >= 0) SelectGroup(m_Design.groups[n].id);
            return;
        }
        std::vector<int> made;
        for (int id : m_Sel.shapes) {
            int s = m_Design.DuplicateShape(m_Design.ShapeIndex(id), { 10, 0, 0 });
            if (s >= 0) made.push_back(m_Design.shapes[s].id);
        }
        if (!made.empty()) {
            m_Sel = Selection{};
            m_Sel.shapes = made;
        }
    }

    void App::ToggleCut() {
        for (int id : m_Sel.shapes) {
            int s = m_Design.ShapeIndex(id);
            if (s < 0) continue;
            Shape& sh = m_Design.shapes[s];
            sh.op = sh.op == Op::Cut ? Op::Union : Op::Cut;
            int g = m_Design.GroupIndex(sh.group);
            sh.color = sh.op == Op::Cut ? Color{ 225, 90, 90, 255 } : (g >= 0 ? m_Design.groups[g].color : WHITE);
        }
    }

    // ---- visibility (view state only: hidden parts still export and still count for collisions) -------------------

    bool App::GroupHidden(int groupId) const {
        int g = groupId, guard = 0;
        while (g >= 0 && guard++ <= (int)m_Design.groups.size()) {
            if (m_Hidden.count(g)) return true;
            int i = m_Design.GroupIndex(g);
            g = i >= 0 ? m_Design.groups[i].parent : -1;
        }
        return false;
    }

    bool App::ShapeHidden(int s) const {
        return m_Hidden.count(m_Design.shapes[s].id) || GroupHidden(m_Design.shapes[s].group);
    }

    void App::ToggleHidden(int id) {
        if (m_Hidden.erase(id)) return;
        m_Hidden.insert(id);
        // whatever just went out of sight drops out of the selection
        if (m_Sel.group >= 0 && GroupHidden(m_Sel.group)) m_Sel.group = -1;
        std::vector<int> keep;
        for (int s : m_Sel.shapes) {
            int i = m_Design.ShapeIndex(s);
            if (i >= 0 && !ShapeHidden(i)) keep.push_back(s);
        }
        m_Sel.shapes = keep;
    }

    void App::HideSelection() {
        if (m_Sel.group >= 0) m_Hidden.insert(m_Sel.group);
        for (int id : m_Sel.shapes) m_Hidden.insert(id);
        if (!m_Sel.Empty()) m_Status = "Hidden (Alt+H shows everything again; hidden parts still export).";
        ClearSelection();
    }

    int App::AddHingeBetween(int parent, int child) {
        if (!m_Design.CanHinge(parent, child)) {
            m_Status = "Can't hinge those: pick two different bodies, the child must not already hang from a hinge, "
                       "and the parent must not hang from the child.";
            return -1;
        }
        int h = m_Design.AddHinge(parent, child);
        PlaceHinge(h);
        SelectHinge(m_Design.hinges[h].id);
        m_Status = "Added " + m_Design.hinges[h].name + ". It is placed where the bodies meet; move it with the gizmo and "
                   "set its open and closed angles.";
        return h;
    }

    // Puts a new hinge on the edge where the two bodies meet, with the axis along that edge, closed in the design pose
    // and opening away from the parent.
    void App::PlaceHinge(int hi) {
        Hinge& h = m_Design.hinges[hi];
        BodyField parent = BodyField::Body(m_Design, h.parent);
        BodyField child = BodyField::Body(m_Design, h.child);
        std::vector<Vector3> pts;
        std::vector<float> dist;
        for (int s : m_Design.GroupShapes(h.child)) {
            const Shape& sh = m_Design.shapes[s];
            ShapeModel& sm = ShapeMesh(sh);
            Rigid pose = sh.sf.Pose();
            size_t stride = std::max<size_t>(1, sm.mesh.positions.size() / 1500);
            for (size_t k = 0; k < sm.mesh.positions.size(); k += stride) {
                Vector3 p = pose.Apply(sm.mesh.positions[k]);
                if (fabsf(child.Distance(p)) > 1.0f) continue;   // only the child's real surface
                pts.push_back(p);
                dist.push_back(parent.Distance(p));
            }
        }
        BoundingBox pb = m_Design.BodyBounds(h.parent), cb = m_Design.BodyBounds(h.child);
        Vector3 pc = Vector3Scale(Vector3Add(pb.min, pb.max), 0.5f), cc = Vector3Scale(Vector3Add(cb.min, cb.max), 0.5f);
        Vector3 offset = Vector3Subtract(cc, pc);
        Vector3 od = Vector3Length(offset) > 1e-4f ? Vector3Normalize(offset) : Vector3{ 0, 1, 0 };
        if (pts.empty()) {
            h.pivot = Vector3Lerp(pc, cc, 0.5f);
            h.axis = Utils::AnyPerpendicular(od);
            return;
        }
        float dmin = *std::min_element(dist.begin(), dist.end());
        float extent = Vector3Distance(cb.min, cb.max);
        std::vector<Vector3> contact;
        for (size_t i = 0; i < pts.size(); i++)
            if (dist[i] < dmin + fmaxf(1.0f, extent * 0.03f)) contact.push_back(pts[i]);
        Vector3 mean = { 0, 0, 0 };
        for (const Vector3& p : contact) mean = Vector3Add(mean, p);
        mean = Vector3Scale(mean, 1.0f / contact.size());
        // principal direction of the contact patch (power iteration on its covariance)
        float cov[3][3] = {};
        for (const Vector3& p : contact) {
            Vector3 q = Vector3Subtract(p, mean);
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++) cov[a][b] += (&q.x)[a] * (&q.x)[b];
        }
        Vector3 axis = { 1, 0.1f, 0.05f };
        for (int it = 0; it < 30; it++) {
            Vector3 n = { cov[0][0] * axis.x + cov[0][1] * axis.y + cov[0][2] * axis.z,
                          cov[1][0] * axis.x + cov[1][1] * axis.y + cov[1][2] * axis.z,
                          cov[2][0] * axis.x + cov[2][1] * axis.y + cov[2][2] * axis.z };
            if (Vector3Length(n) < 1e-9f) break;
            axis = Vector3Normalize(n);
        }
        axis = Vector3Subtract(axis, Vector3Scale(od, Vector3DotProduct(axis, od)));
        if (Vector3Length(axis) < 0.2f) {
            // no clear edge: the world axis most across the offset
            const Vector3 world[3] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
            float best = 2.0f;
            for (const Vector3& w : world)
                if (fabsf(Vector3DotProduct(w, od)) < best) { best = fabsf(Vector3DotProduct(w, od)); axis = w; }
        }
        axis = Vector3Normalize(axis);
        for (int a = 0; a < 3; a++)
            if (fabsf((&axis.x)[a]) > cosf(20.0f * DEG2RAD)) { axis = { 0, 0, 0 }; (&axis.x)[a] = 1.0f; }
        Vector3 across = Vector3CrossProduct(axis, od);
        across = Vector3Length(across) > 1e-4f ? Vector3Normalize(across) : Utils::AnyPerpendicular(axis);
        // the far edge of the patch across the axis, pushed out by the knuckle radius
        Vector3 edge = contact[0];
        float lo = 1e30f, alo = 1e30f, ahi = -1e30f;
        for (const Vector3& p : contact) {
            float e = Vector3DotProduct(p, across), a = Vector3DotProduct(Vector3Subtract(p, mean), axis);
            if (e < lo) { lo = e; edge = p; }
            alo = fminf(alo, a);
            ahi = fmaxf(ahi, a);
        }
        Vector3 pivot = Vector3Subtract(edge, Vector3Scale(across, h.radius * 0.9f));
        pivot = Vector3Subtract(pivot, Vector3Scale(od, fmaxf(dmin, 0.0f) * 0.5f));
        pivot = Vector3Add(pivot, Vector3Scale(axis, Vector3DotProduct(Vector3Subtract(mean, pivot), axis)));
        h.pivot = pivot;
        h.axis = axis;
        h.length = Clamp((ahi - alo) * 0.7f, 8.0f, 80.0f);
        h.knuckles = std::clamp(((int)roundf(h.length / 8.0f)) | 1, 3, 9);
        // open away from the parent
        Vector3 turned = Rigid::AboutAxis(pivot, axis, 10.0f * DEG2RAD).Apply(cc);
        float sign = Vector3Distance(turned, pc) > Vector3Distance(cc, pc) ? 1.0f : -1.0f;
        h.start = sign * 100.0f * DEG2RAD;
        h.end = 0.0f;
    }

    void App::FrameView(Viewport& vp) {
        BoundingBox b = m_Design.Bounds();
        for (const Hinge& h : m_Design.hinges) {
            b.min = Vector3Min(b.min, Vector3SubtractValue(h.pivot, h.radius));
            b.max = Vector3Max(b.max, Vector3AddValue(h.pivot, h.radius));
        }
        if (m_HasReference && m_ShowReference && (&vp == &m_DesignView || m_ReferenceInOutput)) {
            BoundingBox r = ReferenceBounds();
            if (m_Design.shapes.empty()) b = r;
            else { b.min = Vector3Min(b.min, r.min); b.max = Vector3Max(b.max, r.max); }
        }
        if (&vp == &m_OutputView && !m_Design.shapes.empty()) {
            // everything the parts sweep through between open and closed
            for (float t : { 0.0f, 0.5f, 1.0f }) {
                std::vector<Rigid> xf = m_Design.Transforms(t, m_Ease);
                for (int g : m_Design.Bodies()) {
                    BoundingBox bb = m_Design.BodyBounds(m_Design.groups[g].id);
                    for (int i = 0; i < 8; i++) {
                        Vector3 c = { (i & 1) ? bb.max.x : bb.min.x, (i & 2) ? bb.max.y : bb.min.y, (i & 4) ? bb.max.z : bb.min.z };
                        Vector3 w = xf[g].Apply(c);
                        b.min = Vector3Min(b.min, w);
                        b.max = Vector3Max(b.max, w);
                    }
                }
            }
        }
        vp.Frame(b);
    }

    // ---- exact rendering ---------------------------------------------------------------------------------------------

    // Keeps one raymarching program per body (and separate pin) in step with the design. Values change every frame of
    // a drag and only cost an upload; a shader is recompiled only when a body's structure changes.
    void App::SyncExact() {
        if (!Exact()) return;
        if (m_ExactSynced && SameGeometry(m_ExactFor, m_Design)) return;
        std::vector<std::string> slots;
        for (int g : m_Design.Bodies()) {
            BodyField f = BodyField::Body(m_Design, m_Design.groups[g].id);
            if (f.Empty()) continue;
            std::string slot = "b" + std::to_string(m_Design.groups[g].id);
            if (!m_Sdf.SetBody(slot, f)) m_Status = "A body's raymarching shader failed to compile; showing meshes for it.";
            slots.push_back(slot);
        }
        for (int h = 0; h < (int)m_Design.hinges.size(); h++) {
            if (!m_Design.hinges[h].hardware || m_Design.hinges[h].pin != PinStyle::Separate) continue;
            std::string slot = "p" + std::to_string(m_Design.hinges[h].id);
            m_Sdf.SetBody(slot, BodyField::Pin(m_Design, h));
            slots.push_back(slot);
        }
        m_Sdf.Retain(slots);
        m_ExactFor = m_Design;
        m_ExactSynced = true;
    }

    SdfView App::ViewOf(const Viewport& vp) const {
        SdfView v;
        v.camera = vp.GetCamera();
        v.ortho = vp.Ortho();
        float h = fmaxf(vp.ScreenRectangle().height, 1.0f);
        v.pixel = v.ortho ? v.camera.fovy / h : 2.0f * tanf(v.camera.fovy * 0.5f * DEG2RAD) / h;
        return v;
    }

    // ---- frame -----------------------------------------------------------------------------------------------------

    void App::UpdateAnimation() {
        float dt = fminf(GetFrameTime(), 0.1f);
        if (m_Anim != 0) {
            m_T = Clamp(m_T + m_Anim * dt / fmaxf(m_Duration, 0.2f), 0.0f, 1.0f);
            if ((m_Anim > 0 && m_T >= 1.0f) || (m_Anim < 0 && m_T <= 0.0f)) {
                m_Anim = 0;
                m_LoopTimer = 0.0f;
            }
        } else if (m_Loop) {
            m_LoopTimer += dt;
            if (m_LoopTimer > 0.6f) {
                m_LoopTimer = 0.0f;
                m_Anim = m_T > 0.5f ? -1 : 1;
            }
        }
    }

    void App::Frame() {
        HandleFileDrops();
        PollJobs();
        UpdateAnimation();
        PurgeShapeMeshes();

        float W = (float)GetScreenWidth(), H = (float)GetScreenHeight();
        float sidebar = fminf(400.0f, W * 0.3f);
        float half = (W - sidebar) * 0.5f;

        rlImGuiBegin();
        ImGuizmo::BeginFrame();
        DrawSidebar(sidebar, H);
        DrawViewportWindow(m_DesignView, ImVec2(sidebar, 0), ImVec2(half, H), true);
        DrawViewportWindow(m_OutputView, ImVec2(sidebar + half, 0), ImVec2(W - sidebar - half, H), false);
        HandleKeys();
        SanitizeSelection();
        TrackHistory();
        UpdateJobs();

        SyncExact();
        RenderDesign();
        RenderOutput();

        BeginDrawing();
        ClearBackground({ 20, 20, 24, 255 });
        rlImGuiEnd();
        EndDrawing();
    }
}
