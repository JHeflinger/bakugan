#include "raylib.h"
#include "imgui.h"
#include "imgui_internal.h"   // ImRect, BeginDragDropTargetCustom
#include "rlImGui.h"
#include "ImGuizmo.h"
#include "viewport.hpp"
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <string>
#include <vector>

static const char* kLitVS = R"(#version 330
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;
uniform mat4 mvp;
uniform mat4 matModel;
uniform mat4 matNormal;
out vec3 fragPos;
out vec2 fragTexCoord;
out vec3 fragNormal;
out vec4 fragColor;
void main() {
    fragPos = vec3(matModel * vec4(vertexPosition, 1.0));
    fragTexCoord = vertexTexCoord;
    fragColor = vertexColor;
    fragNormal = vec3(matNormal * vec4(vertexNormal, 0.0));
    gl_Position = mvp * vec4(vertexPosition, 1.0);
})";

static const char* kLitFS = R"(#version 330
in vec3 fragPos;
in vec2 fragTexCoord;
in vec3 fragNormal;
in vec4 fragColor;
uniform sampler2D texture0;
uniform vec4 colDiffuse;
uniform vec3 viewPos;
out vec4 finalColor;
void main() {
    vec4 base = texture(texture0, fragTexCoord) * colDiffuse * fragColor;
    float light = 1.0;
    if (length(fragNormal) > 0.01) {
        vec3 n = normalize(fragNormal);
        if (dot(n, viewPos - fragPos) < 0.0) n = -n;   // two-sided
        vec3 key = normalize(vec3(0.4, 1.0, 0.6));
        light = 0.35 + 0.65 * max(dot(n, key), 0.0);
    }
    finalColor = vec4(base.rgb * light, base.a);
})";

// ---------------------------------------------------------------- asset library

enum class Primitive { None, Cube, Sphere, Cylinder, Torus, Knot };

struct Asset {
    std::string name;
    std::string path;                 // empty for primitives
    Primitive prim = Primitive::None;
};

static bool IsModelFile(const char* path) {
    return IsFileExtension(path, ".obj;.gltf;.glb;.iqm;.vox;.m3d");
}

static bool LoadAsset(const Asset& a, Model* out, std::string* err) {
    Mesh mesh = {};
    switch (a.prim) {
        case Primitive::Cube:     mesh = GenMeshCube(1, 1, 1); break;
        case Primitive::Sphere:   mesh = GenMeshSphere(0.5f, 24, 24); break;
        case Primitive::Cylinder: mesh = GenMeshCylinder(0.3f, 1.5f, 24); break;
        case Primitive::Torus:    mesh = GenMeshTorus(0.3f, 1.0f, 24, 32); break;
        case Primitive::Knot:     mesh = GenMeshKnot(0.5f, 1.5f, 64, 128); break;
        case Primitive::None: {
            if (!FileExists(a.path.c_str())) { *err = "File not found: " + a.path; return false; }
            Model m = LoadModel(a.path.c_str());
            if (m.meshCount == 0 || !IsModelValid(m)) {
                UnloadModel(m);
                *err = "Could not load " + a.name;
                return false;
            }
            *out = m;
            return true;
        }
    }
    *out = LoadModelFromMesh(mesh);
    return true;
}

struct App {
    std::vector<Asset> library;
    Viewport left{ "Rig  (edit skeleton)", true };
    Viewport right{ "Reference  (view only)", false };
    Shader lit = {};
    int viewPosLoc = -1;
    std::string status = "Drag a model file onto either panel to begin.";

    // rotation editor state for the selected joint (Euler angles are only a UI view of the quaternion)
    int eulerJoint = -1;
    Quaternion eulerQuat = {};
    Vector3 eulerDeg = {};

    int AddToLibrary(const Asset& a) {
        for (int i = 0; i < (int)library.size(); i++)
            if (!a.path.empty() && library[i].path == a.path) return i;
        library.push_back(a);
        return (int)library.size() - 1;
    }

    void LoadInto(Viewport& vp, int assetIndex) {
        const Asset& a = library[assetIndex];
        Model m;
        std::string err;
        if (!LoadAsset(a, &m, &err)) { status = err; return; }
        vp.SetModel(m, a.name, lit);
        status = "Loaded " + a.name + " into " + (vp.rigEditable ? "left" : "right") + " panel.";
    }

    void HandleFileDrops() {
        if (!IsFileDropped()) return;
        FilePathList files = LoadDroppedFiles();
        Vector2 m = GetMousePosition();
        Viewport* vp = CheckCollisionPointRec(m, right.screenRect) ? &right
                     : CheckCollisionPointRec(m, left.screenRect) ? &left
                     : nullptr;
        int toLoad = -1, skipped = 0;
        for (unsigned i = 0; i < files.count; i++) {
            const char* path = files.paths[i];
            if (!IsModelFile(path)) { skipped++; continue; }
            int idx = AddToLibrary({ GetFileName(path), path });
            if (toLoad < 0) toLoad = idx;
        }
        UnloadDroppedFiles(files);

        if (toLoad >= 0 && vp) LoadInto(*vp, toLoad);
        else if (toLoad >= 0) status = "Added to library. Drag it onto a panel to load it.";
        if (skipped) status += " (" + std::to_string(skipped) + " unsupported file(s) ignored)";
    }

    void AcceptAssetDrop(Viewport& vp) {
        Rectangle r = vp.screenRect;
        ImRect rect(r.x, r.y, r.x + r.width, r.y + r.height);
        if (!ImGui::BeginDragDropTargetCustom(rect, ImGui::GetID("##viewport_drop"))) return;
        if (const ImGuiPayload* p = ImGui::AcceptDragDropPayload("ASSET"))
            LoadInto(vp, *(const int*)p->Data);
        ImGui::EndDragDropTarget();
    }

    void DrawViewportWindow(Viewport& vp, ImVec2 pos, ImVec2 size, bool focusHint) {
        ImGui::SetNextWindowPos(pos);
        ImGui::SetNextWindowSize(size);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(4, 4));
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
                                 ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
        if (focusHint) ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.25f, 0.18f, 0.08f, 1));
        ImGui::Begin(vp.title.c_str(), nullptr, flags);
        if (focusHint) ImGui::PopStyleColor();
        vp.DrawPanel();
        AcceptAssetDrop(vp);
        ImGui::End();
        ImGui::PopStyleVar();
    }

    void DrawLibrary() {
        ImGui::SeparatorText("Model library");
        ImGui::TextDisabled("Drag an item onto a panel, or drop files\nfrom Finder straight onto a panel.");
        for (int i = 0; i < (int)library.size(); i++) {
            ImGui::PushID(i);
            ImGui::Selectable(library[i].name.c_str());
            if (!library[i].path.empty()) ImGui::SetItemTooltip("%s", library[i].path.c_str());
            if (ImGui::BeginDragDropSource()) {
                ImGui::SetDragDropPayload("ASSET", &i, sizeof(int));
                ImGui::Text("Load %s", library[i].name.c_str());
                ImGui::EndDragDropSource();
            }
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Load into left panel")) LoadInto(left, i);
                if (ImGui::MenuItem("Load into right panel")) LoadInto(right, i);
                ImGui::EndPopup();
            }
            ImGui::PopID();
        }
    }

    // ImGui wants float colours; the skeleton keeps raylib's byte Color.
    static bool ColorEdit(const char* id, Color* c) {
        float f[3] = { c->r / 255.0f, c->g / 255.0f, c->b / 255.0f };
        if (!ImGui::ColorEdit3(id, f, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel)) return false;
        *c = { (unsigned char)(f[0] * 255), (unsigned char)(f[1] * 255), (unsigned char)(f[2] * 255), 255 };
        return true;
    }

    void DrawJointTree(int index) {
        Skeleton& s = left.skeleton;
        bool leaf = true;
        for (const Joint& j : s.joints) if (j.parent == index) { leaf = false; break; }
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_DefaultOpen |
                                   ImGuiTreeNodeFlags_SpanAvailWidth;
        if (leaf) flags |= ImGuiTreeNodeFlags_Leaf;
        if (index == left.selJoint) flags |= ImGuiTreeNodeFlags_Selected;
        bool open = ImGui::TreeNodeEx((void*)(intptr_t)(index + 1), flags, "%s", s.joints[index].name.c_str());
        if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) left.selJoint = index;
        if (ImGui::BeginPopupContextItem()) {
            left.selJoint = index;
            if (ImGui::MenuItem("Add child bone")) left.AddChildBone(index);
            if (ImGui::MenuItem("Delete (with children)", nullptr, false, index != 0)) left.DeleteSelectedJoint();
            ImGui::EndPopup();
        }
        if (open) {
            for (int i = index + 1; i < (int)s.joints.size(); i++)
                if (s.joints[i].parent == index) DrawJointTree(i);
            ImGui::TreePop();
        }
    }

    void DrawHierarchy() {
        Skeleton& s = left.skeleton;
        ImGui::SeparatorText("Skeleton hierarchy (left panel)");
        ImGui::Checkbox("##vis", &s.visible);
        ImGui::SetItemTooltip("Show skeleton");
        ImGui::SameLine();
        ColorEdit("##col", &s.color);
        ImGui::SameLine();
        ImGui::Text("%d joints", (int)s.joints.size());

        int parent = left.selJoint >= 0 ? left.selJoint : 0;
        std::string addLabel = "Add child bone to '" + s.joints[parent].name + "'  (C)";
        if (ImGui::Button(addLabel.c_str(), ImVec2(-FLT_MIN, 0))) left.AddChildBone(parent);

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
                Color c = s.color;
                s = MakeHumanoidSkeleton(left.bounds);
                s.color = c;
                left.selJoint = 0;
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
        if (ImGui::BeginPopup("confirm_clear")) {
            ImGui::Text("Remove every bone, keeping only the root?");
            if (ImGui::Button("Clear")) { left.ResetSkeleton(); ImGui::CloseCurrentPopup(); }
            ImGui::EndPopup();
        }
    }

    void DrawJointInspector() {
        Skeleton& s = left.skeleton;
        int sel = left.selJoint;
        ImGui::SeparatorText("Selected joint");
        if (sel < 0 || sel >= (int)s.joints.size()) {
            ImGui::TextDisabled("Click a joint in the left panel to select it.");
            return;
        }
        Joint& j = s.joints[sel];

        char name[64];
        snprintf(name, sizeof(name), "%s", j.name.c_str());
        if (ImGui::InputText("Name", name, sizeof(name))) j.name = name;
        ImGui::Text("Parent: %s", j.parent >= 0 ? s.joints[j.parent].name.c_str() : "(root of the hierarchy)");

        std::vector<Matrix> world = s.WorldTransforms();
        Vector3 wp = MatrixPosition(world[sel]);
        if (ImGui::DragFloat3("Position", &wp.x, 0.005f, 0, 0, "%.3f")) left.MoveJoint(sel, wp);
        ImGui::SetItemTooltip("World-space position. Respects 'Move subtree'.");

        // ball joint rotation, edited as Euler angles
        if (eulerJoint != sel || !QuaternionEquals(eulerQuat, j.rotation)) {
            eulerDeg = Vector3Scale(QuaternionToEuler(j.rotation), RAD2DEG);
            eulerJoint = sel;
            eulerQuat = j.rotation;
        }
        if (ImGui::DragFloat3("Rotation", &eulerDeg.x, 0.5f, -180, 180, "%.1f deg")) {
            Quaternion q = QuaternionFromEuler(eulerDeg.x * DEG2RAD, eulerDeg.y * DEG2RAD, eulerDeg.z * DEG2RAD);
            j.rotation = ClampBallJoint(q, j.swingLimit, j.twistLimit);
            if (!QuaternionEquals(q, j.rotation)) eulerDeg = Vector3Scale(QuaternionToEuler(j.rotation), RAD2DEG);
            eulerQuat = j.rotation;
        }
        bool limits = ImGui::SliderFloat("Swing limit", &j.swingLimit, 0, 180, "%.0f deg");
        ImGui::SetItemTooltip("Cone half-angle around the joint's local +Y axis");
        limits |= ImGui::SliderFloat("Twist limit", &j.twistLimit, 0, 180, "+/- %.0f deg");
        ImGui::SetItemTooltip("Rotation allowed about the joint's local +Y axis");
        if (limits) j.rotation = ClampBallJoint(j.rotation, j.swingLimit, j.twistLimit);

        if (ImGui::Button("Reset rotation")) j.rotation = QuaternionIdentity();
        ImGui::SameLine();
        ImGui::BeginDisabled(sel == 0);
        if (ImGui::Button("Delete (with children)")) left.DeleteSelectedJoint();
        ImGui::EndDisabled();
        if (sel == 0) ImGui::SetItemTooltip("The root can't be deleted");
    }

    void DrawSidebar(float width, float height) {
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(width, height));
        ImGui::Begin("Inspector", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                           ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings);
        DrawLibrary();
        DrawHierarchy();
        DrawJointInspector();
        ImGui::SeparatorText("Status");
        ImGui::TextWrapped("%s", status.c_str());
        ImGui::End();
    }

    void Frame() {
        HandleFileDrops();

        float W = (float)GetScreenWidth(), H = (float)GetScreenHeight();
        float sidebar = fminf(360.0f, W * 0.3f);
        float half = (W - sidebar) * 0.5f;

        rlImGuiBegin();
        ImGuizmo::BeginFrame();
        DrawSidebar(sidebar, H);
        DrawViewportWindow(left, ImVec2(sidebar, 0), ImVec2(half, H), true);
        DrawViewportWindow(right, ImVec2(sidebar + half, 0), ImVec2(W - sidebar - half, H), false);

        left.Render(lit, viewPosLoc);
        right.Render(lit, viewPosLoc);

        BeginDrawing();
        ClearBackground({ 20, 20, 24, 255 });
        rlImGuiEnd();
        EndDrawing();
    }
};

int main() {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_WINDOW_HIGHDPI | FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
    InitWindow(1280, 720, "Rig Editor");
    {   // fit the window to the monitor (a too-large request gets clipped by the OS)
        int mon = GetCurrentMonitor();
        int mw = GetMonitorWidth(mon), mh = GetMonitorHeight(mon);
        int w = std::min(1600, (int)(mw * 0.9f)), h = std::min(950, (int)(mh * 0.85f));
        SetWindowSize(w, h);
        SetWindowPosition((mw - w) / 2, (mh - h) / 2);
    }
    SetExitKey(KEY_NULL);
    SetTargetFPS(60);
    rlImGuiSetup(true);
    ImGui::GetIO().IniFilename = nullptr;

    {
        App app;
        app.lit = LoadShaderFromMemory(kLitVS, kLitFS);
        app.viewPosLoc = GetShaderLocation(app.lit, "viewPos");
        app.AddToLibrary({ "Cube", "", Primitive::Cube });
        app.AddToLibrary({ "Sphere", "", Primitive::Sphere });
        app.AddToLibrary({ "Cylinder", "", Primitive::Cylinder });
        app.AddToLibrary({ "Torus", "", Primitive::Torus });
        app.AddToLibrary({ "Knot", "", Primitive::Knot });

        while (!WindowShouldClose()) app.Frame();

        app.left.Unload();
        app.right.Unload();
        UnloadShader(app.lit);
    }

    rlImGuiShutdown();
    CloseWindow();
    return 0;
}
