#include "app.hpp"
#include "util/utils.hpp"
#include <imgui_internal.h>
#include <rlImGui.h>
#include <ImGuizmo.h>
#include <algorithm>
#include <raymath.h>
#include <raylib.h>
#include <cstdint>
#include <imgui.h>
#include <cfloat>
#include <cstdio>
#include <string>
#include <vector>
#include <cmath>

#define MAX_UNDO 200

namespace Bakugan {
    App::App() {
        m_Right.SetMirror(&m_Left);
        m_Committed = Capture();
	    SetTraceLogLevel(LOG_NONE);
        SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_WINDOW_HIGHDPI | FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
        InitWindow(1280, 720, "Bakugan Everything");
        int mon = GetCurrentMonitor();
        int mw = GetMonitorWidth(mon), mh = GetMonitorHeight(mon);
        int w = std::min(1600, (int)(mw * 0.9f)), h = std::min(950, (int)(mh * 0.85f));
        SetWindowSize(w, h);
        SetWindowPosition((mw - w) / 2, (mh - h) / 2);
        SetExitKey(KEY_NULL);
        SetTargetFPS(60);
        rlImGuiSetup(true);
        ImGui::GetIO().IniFilename = nullptr;
        m_Shader = LoadShaderFromMemory(Utils::GetVertexShader(), Utils::GetFragmentShader());
        m_ViewPositionLocation = GetShaderLocation(m_Shader, "viewPos");
        AddToLibrary({ "Cube", "", Primitive::Cube });
        AddToLibrary({ "Sphere", "", Primitive::Sphere });
        AddToLibrary({ "Cylinder", "", Primitive::Cylinder });
        AddToLibrary({ "Torus", "", Primitive::Torus });
        AddToLibrary({ "Knot", "", Primitive::Knot });
    }

    App::~App() {
        m_Left.Unload();
        m_Right.Unload();
        UnloadShader(m_Shader);
        rlImGuiShutdown();
        CloseWindow();
    }

    void App::Run() {
        while (!WindowShouldClose()) Frame();
    }

    EditState App::Capture() const {
        return { m_Left.GetSkeleton(), m_Left.SelectedJoint(), m_Right.PoseRotations(), m_Right.RootOffset(), m_Right.SelectedJoint() };
    }

    void App::Restore(const EditState& st) {
        m_Left.SkeletonRef() = st.skeleton;
        m_Left.SelectJoint(st.leftSel < (int)st.skeleton.Joints().size() ? st.leftSel : 0);
        m_Right.PoseRotations() = st.pose;
        m_Right.SetRootOffset(st.offset);
        m_Right.SelectJoint(st.rightSel < (int)st.skeleton.Joints().size() ? st.rightSel : -1);
    }

    bool App::Interacting() {
        return ImGui::IsAnyItemActive() || ImGuizmo::IsUsingAny() || ImGui::IsMouseDown(ImGuiMouseButton_Left);
    }

    void App::TrackHistory() {
        if (Interacting()) return;
        EditState cur = Capture();
        if (Utils::SameEdit(cur, m_Committed)) {   // keep the selection current, so undo restores it too
            m_Committed.leftSel = cur.leftSel;
            m_Committed.rightSel = cur.rightSel;
            return;
        }
        m_UndoStack.push_back(std::move(m_Committed));
        if (m_UndoStack.size() > MAX_UNDO) m_UndoStack.erase(m_UndoStack.begin());
        m_RedoStack.clear();
        m_Committed = std::move(cur);
    }

    void App::Undo() {
        TrackHistory();   // commit anything still pending first
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

    void App::HandleUndoKeys() {
        ImGuiIO& io = ImGui::GetIO();
        if (io.WantTextInput || !(io.KeyCtrl || io.KeySuper)) return;   // text fields have their own undo
        if (ImGui::IsKeyPressed(ImGuiKey_Z)) { if (io.KeyShift) Redo(); else Undo(); }
        if (ImGui::IsKeyPressed(ImGuiKey_Y)) Redo();
    }
    
    int App::AddToLibrary(const Asset& a) {
        for (int i = 0; i < (int)m_Library.size(); i++)
            if (!a.path.empty() && m_Library[i].path == a.path) return i;
        m_Library.push_back(a);
        return (int)m_Library.size() - 1;
    }

    void App::LoadInto(Viewport& vp, int assetIndex) {
        const Asset& a = m_Library[assetIndex];
        Model m;
        std::string err;
        if (!Utils::LoadAsset(a, &m, &err)) { m_Status = err; return; }
        vp.SetModel(m, a.name, m_Shader);
        m_Status = "Loaded " + a.name + " into " + (vp.Editable() ? "left" : "right") + " panel.";
    }

    void App::HandleFileDrops() {
        if (!IsFileDropped()) return;
        FilePathList files = LoadDroppedFiles();
        Vector2 m = GetMousePosition();
        Viewport* vp = CheckCollisionPointRec(m, m_Right.ScreenRectangle()) ? &m_Right
                     : CheckCollisionPointRec(m, m_Left.ScreenRectangle()) ? &m_Left
                     : nullptr;
        int toLoad = -1, skipped = 0;
        for (unsigned i = 0; i < files.count; i++) {
            const char* path = files.paths[i];
            if (!Utils::IsModelFile(path)) { skipped++; continue; }
            int idx = AddToLibrary({ GetFileName(path), path });
            if (toLoad < 0) toLoad = idx;
        }
        UnloadDroppedFiles(files);

        if (toLoad >= 0 && vp) LoadInto(*vp, toLoad);
        else if (toLoad >= 0) m_Status = "Added to library. Drag it onto a panel to load it.";
        if (skipped) m_Status += " (" + std::to_string(skipped) + " unsupported file(s) ignored)";
    }

    void App::AcceptAssetDrop(Viewport& vp) {
        Rectangle r = vp.ScreenRectangle();
        ImRect rect(r.x, r.y, r.x + r.width, r.y + r.height);
        if (!ImGui::BeginDragDropTargetCustom(rect, ImGui::GetID("##viewport_drop"))) return;
        if (const ImGuiPayload* p = ImGui::AcceptDragDropPayload("ASSET"))
            LoadInto(vp, *(const int*)p->Data);
        ImGui::EndDragDropTarget();
    }

    void App::DrawViewportWindow(Viewport& vp, ImVec2 pos, ImVec2 size, bool focusHint) {
        ImGui::SetNextWindowPos(pos);
        ImGui::SetNextWindowSize(size);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(4, 4));
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
                                 ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
        if (focusHint) ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.25f, 0.18f, 0.08f, 1));
        ImGui::Begin(vp.Title().c_str(), nullptr, flags);
        if (focusHint) ImGui::PopStyleColor();
        vp.DrawPanel();
        AcceptAssetDrop(vp);
        ImGui::End();
        ImGui::PopStyleVar();
    }

    void App::DrawLibrary() {
        ImGui::SeparatorText("Model library");
        ImGui::TextDisabled("Drag an item onto a panel, or drop files\nfrom Finder straight onto a panel.");
        for (int i = 0; i < (int)m_Library.size(); i++) {
            ImGui::PushID(i);
            ImGui::Selectable(m_Library[i].name.c_str());
            if (!m_Library[i].path.empty()) ImGui::SetItemTooltip("%s", m_Library[i].path.c_str());
            if (ImGui::BeginDragDropSource()) {
                ImGui::SetDragDropPayload("ASSET", &i, sizeof(int));
                ImGui::Text("Load %s", m_Library[i].name.c_str());
                ImGui::EndDragDropSource();
            }
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Load into left panel")) LoadInto(m_Left, i);
                if (ImGui::MenuItem("Load into right panel")) LoadInto(m_Right, i);
                ImGui::EndPopup();
            }
            ImGui::PopID();
        }
    }

    // ImGui wants float colours; the skeleton keeps raylib's byte Color.
    bool App::ColorEdit(const char* id, Color* c) {
        float f[3] = { c->r / 255.0f, c->g / 255.0f, c->b / 255.0f };
        if (!ImGui::ColorEdit3(id, f, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel)) return false;
        *c = { (unsigned char)(f[0] * 255), (unsigned char)(f[1] * 255), (unsigned char)(f[2] * 255), 255 };
        return true;
    }

    void App::DrawJointTree(int index) {
        Skeleton& s = m_Left.SkeletonRef();
        bool leaf = true;
        for (const Joint& j : s.Joints()) if (j.parent == index) { leaf = false; break; }
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_DefaultOpen |
                                   ImGuiTreeNodeFlags_SpanAvailWidth;
        if (leaf) flags |= ImGuiTreeNodeFlags_Leaf;
        if (index == m_Left.SelectedJoint()) flags |= ImGuiTreeNodeFlags_Selected;
        bool open = ImGui::TreeNodeEx((void*)(intptr_t)(index + 1), flags, "%s", s.Joints()[index].name.c_str());
        if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) m_Left.SelectJoint(index);
        if (ImGui::BeginPopupContextItem()) {
            m_Left.SelectJoint(index);
            if (ImGui::MenuItem("Add child bone")) m_Left.AddChildBone(index);
            if (ImGui::MenuItem("Split bone in 2", nullptr, false, index != 0)) {
                m_Left.SetSplitSegments(2);
                m_Left.SplitSelectedBone();
            }
            if (ImGui::MenuItem("Delete (with children)", nullptr, false, index != 0)) m_Left.DeleteSelectedJoint();
            ImGui::EndPopup();
        }
        if (open) {
            for (int i = index + 1; i < (int)s.Joints().size(); i++)
                if (s.Joints()[i].parent == index) DrawJointTree(i);
            ImGui::TreePop();
        }
    }

    void App::DrawHierarchy() {
        Skeleton& s = m_Left.SkeletonRef();
        ImGui::SeparatorText("Edit");
        ImGui::BeginDisabled(m_UndoStack.empty());
        if (ImGui::Button("Undo")) Undo();
        ImGui::EndDisabled();
        ImGui::SetItemTooltip("Cmd/Ctrl+Z  (%d steps)", (int)m_UndoStack.size());
        ImGui::SameLine();
        ImGui::BeginDisabled(m_RedoStack.empty());
        if (ImGui::Button("Redo")) Redo();
        ImGui::EndDisabled();
        ImGui::SetItemTooltip("Cmd/Ctrl+Shift+Z or Ctrl+Y  (%d steps)", (int)m_RedoStack.size());

        ImGui::SeparatorText("Skeleton hierarchy (left panel)");
        bool v = s.IsVisible();
        ImGui::Checkbox("##vis", &v);
        s.SetVisibility(v);
        ImGui::SetItemTooltip("Show skeleton");
        ImGui::SameLine();
        Color tc = s.GetColor();
        ColorEdit("##col", &tc);
        s.SetColor(tc);
        ImGui::SameLine();
        ImGui::Text("%d Joints()", (int)s.Joints().size());

        int parent = m_Left.SelectedJoint() >= 0 ? m_Left.SelectedJoint() : 0;
        std::string addLabel = "Add child bone to '" + s.Joints()[parent].name + "'  (C)";
        if (ImGui::Button(addLabel.c_str(), ImVec2(-FLT_MIN, 0))) m_Left.AddChildBone(parent);

        if (ImGui::BeginChild("joint_tree", ImVec2(0, 200), ImGuiChildFlags_Borders))
            DrawJointTree(0);   // joint 0 is always the single root
        ImGui::EndChild();

        if (ImGui::Button("Reset pose")) s.ResetPose();
        ImGui::SetItemTooltip("Set every ball joint back to its rest rotation");
        ImGui::SameLine();
        if (ImGui::Button("Humanoid template")) ImGui::OpenPopup("confirm_humanoid");
        ImGui::SameLine();
        if (ImGui::Button("Clear bones")) ImGui::OpenPopup("confirm_clear");

        if (ImGui::BeginPopup("confirm_humanoid")) {
            ImGui::Text("Replace the skeleton with a humanoid rig?");
            if (ImGui::Button("Replace")) {
                Color c = s.GetColor();
                s = Utils::MakeHumanoidSkeleton(m_Left.Bounds());
                s.SetColor(c);
                m_Left.SelectJoint(0);
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
        if (ImGui::BeginPopup("confirm_clear")) {
            ImGui::Text("Remove every bone, keeping only the root?");
            if (ImGui::Button("Clear")) { m_Left.ResetSkeleton(); ImGui::CloseCurrentPopup(); }
            ImGui::EndPopup();
        }
    }

    void App::DrawJointInspector() {
        Skeleton& s = m_Left.SkeletonRef();
        int sel = m_Left.SelectedJoint();
        ImGui::SeparatorText("Selected joint");
        if (sel < 0 || sel >= (int)s.Joints().size()) {
            ImGui::TextDisabled("Click a joint in the left panel to select it.");
            return;
        }
        Joint& j = s.JointsRef()[sel];

        char name[64];
        snprintf(name, sizeof(name), "%s", j.name.c_str());
        if (ImGui::InputText("Name", name, sizeof(name))) j.name = name;
        ImGui::Text("Parent: %s", j.parent >= 0 ? s.Joints()[j.parent].name.c_str() : "(root of the hierarchy)");

        std::vector<Matrix> world = s.WorldTransforms();
        Vector3 wp = Utils::MatrixPosition(world[sel]);
        if (ImGui::DragFloat3("Position", &wp.x, 0.005f, 0, 0, "%.3f")) m_Left.MoveJoint(sel, wp);
        ImGui::SetItemTooltip("World-space position. Respects 'Move subtree'.");

        // ball joint rotation, edited as Euler angles
        if (m_EulerJoint != sel || !QuaternionEquals(m_EulerQuaternion, j.rotation)) {
            m_EulerDegree = Vector3Scale(QuaternionToEuler(j.rotation), RAD2DEG);
            m_EulerJoint = sel;
            m_EulerQuaternion = j.rotation;
        }
        if (ImGui::DragFloat3("Rotation", &m_EulerDegree.x, 0.5f, -180, 180, "%.1f deg")) {
            Quaternion q = QuaternionFromEuler(m_EulerDegree.x * DEG2RAD, m_EulerDegree.y * DEG2RAD, m_EulerDegree.z * DEG2RAD);
            j.rotation = Utils::ClampBallJoint(q, j.swingLimit, j.twistLimit);
            if (!QuaternionEquals(q, j.rotation)) m_EulerDegree = Vector3Scale(QuaternionToEuler(j.rotation), RAD2DEG);
            m_EulerQuaternion = j.rotation;
        }
        bool limits = ImGui::SliderFloat("Swing limit", &j.swingLimit, 0, 180, "%.0f deg");
        ImGui::SetItemTooltip("Cone half-angle around the joint's local +Y axis");
        limits |= ImGui::SliderFloat("Twist limit", &j.twistLimit, 0, 180, "+/- %.0f deg");
        ImGui::SetItemTooltip("Rotation allowed about the joint's local +Y axis");
        if (limits) j.rotation = Utils::ClampBallJoint(j.rotation, j.swingLimit, j.twistLimit);

        if (ImGui::Button("Reset rotation")) j.rotation = QuaternionIdentity();

        // Split the bone that ends at this joint (parent -> joint). This and Delete change
        // the joint array, so they come last (`j` is not used after them).
        std::string tip = sel == 0 ? "The root has no bone to split"
                                   : "Insert Joints() evenly along the bone from '" + s.Joints()[j.parent].name + "' to this joint";
        ImGui::BeginDisabled(sel == 0);
        ImGui::SetNextItemWidth(90);
        int splitsegs = m_Left.SplitSegments();
        ImGui::InputInt("##pieces", &splitsegs);
        m_Left.SetSplitSegments(splitsegs);
        m_Left.SetSplitSegments(std::clamp(m_Left.SplitSegments(), 2, 16));
        ImGui::SameLine();
        if (ImGui::Button("Split bone into pieces")) m_Left.SplitSelectedBone();
        ImGui::SetItemTooltip("%s", tip.c_str());
        if (ImGui::Button("Delete (with children)")) m_Left.DeleteSelectedJoint();
        ImGui::EndDisabled();
        if (sel == 0) ImGui::SetItemTooltip("The root can't be deleted");
    }

    void App::DrawSidebar(float width, float height) {
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(width, height));
        ImGui::Begin("Inspector", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                           ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings);
        DrawLibrary();
        DrawHierarchy();
        DrawJointInspector();
        ImGui::SeparatorText("Status");
        ImGui::TextWrapped("%s", m_Status.c_str());
        ImGui::End();
    }

    void App::Frame() {
        HandleFileDrops();

        float W = (float)GetScreenWidth(), H = (float)GetScreenHeight();
        float sidebar = fminf(360.0f, W * 0.3f);
        float half = (W - sidebar) * 0.5f;

        rlImGuiBegin();
        ImGuizmo::BeginFrame();
        DrawSidebar(sidebar, H);
        DrawViewportWindow(m_Left, ImVec2(sidebar, 0), ImVec2(half, H), true);
        DrawViewportWindow(m_Right, ImVec2(sidebar + half, 0), ImVec2(W - sidebar - half, H), false);

        HandleUndoKeys();
        TrackHistory();

        m_Left.Render(m_Shader, m_ViewPositionLocation);
        m_Right.Render(m_Shader, m_ViewPositionLocation);

        BeginDrawing();
        ClearBackground({ 20, 20, 24, 255 });
        rlImGuiEnd();
        EndDrawing();
    }

}
