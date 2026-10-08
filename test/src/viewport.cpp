#include "viewport.hpp"
#include "rlgl.h"
#include "ImGuizmo.h"
#include <cfloat>
#include <cmath>
#include <cstring>
#include <map>
#include <tuple>

Viewport::Viewport(std::string t, bool editable) : title(std::move(t)), rigEditable(editable) {
    skeleton.name = "Skeleton";
    ResetSkeleton();
}

void Viewport::Unload() {
    if (texW > 0) UnloadRenderTexture(target);
    ClearModel();
}

// ---------------------------------------------------------------- model

// Some files (e.g. OBJs with only "v" and "f" lines) have no normals; raylib then fills in
// a constant placeholder. Generate smooth ones by averaging face normals over vertices
// that share a position.
static void EnsureNormals(Mesh& mesh) {
    if (!mesh.vertices || mesh.vertexCount < 3) return;
    if (mesh.normals) {
        for (int i = 3; i < mesh.vertexCount * 3; i++)
            if (mesh.normals[i] != mesh.normals[i % 3]) return;   // varying normals: keep them
    } else {
        mesh.normals = (float*)MemAlloc(mesh.vertexCount * 3 * sizeof(float));
    }

    const Vector3* v = (const Vector3*)mesh.vertices;
    int triCount = mesh.indices ? mesh.triangleCount : mesh.vertexCount / 3;
    auto index = [&](int t, int k) { return mesh.indices ? (int)mesh.indices[t * 3 + k] : t * 3 + k; };

    std::map<std::tuple<float, float, float>, Vector3> accum;
    auto key = [&](int i) { return std::make_tuple(v[i].x, v[i].y, v[i].z); };
    for (int t = 0; t < triCount; t++) {
        int a = index(t, 0), b = index(t, 1), c = index(t, 2);
        // area-weighted: the cross product's length is twice the triangle area
        Vector3 n = Vector3CrossProduct(Vector3Subtract(v[b], v[a]), Vector3Subtract(v[c], v[a]));
        for (int i : { a, b, c }) accum[key(i)] = Vector3Add(accum[key(i)], n);
    }
    Vector3* out = (Vector3*)mesh.normals;
    for (int i = 0; i < mesh.vertexCount; i++) out[i] = Vector3Normalize(accum[key(i)]);

    int bytes = mesh.vertexCount * 3 * sizeof(float);
    if (mesh.vaoId == 0) {
        UploadMesh(&mesh, false);
    } else if (mesh.vboId[RL_DEFAULT_SHADER_ATTRIB_LOCATION_NORMAL] != 0) {
        UpdateMeshBuffer(mesh, RL_DEFAULT_SHADER_ATTRIB_LOCATION_NORMAL, mesh.normals, bytes, 0);
    } else {
        // uploaded without a normal buffer: add one to the existing VAO (as UploadMesh does)
        rlEnableVertexArray(mesh.vaoId);
        mesh.vboId[RL_DEFAULT_SHADER_ATTRIB_LOCATION_NORMAL] = rlLoadVertexBuffer(mesh.normals, bytes, false);
        rlSetVertexAttribute(RL_DEFAULT_SHADER_ATTRIB_LOCATION_NORMAL, 3, RL_FLOAT, 0, 0, 0);
        rlEnableVertexAttribute(RL_DEFAULT_SHADER_ATTRIB_LOCATION_NORMAL);
        rlDisableVertexArray();
    }
}

void Viewport::SetModel(Model m, const std::string& name, Shader lit) {
    ClearModel();
    model = m;
    hasModel = true;
    modelName = name;
    for (int i = 0; i < model.materialCount; i++) model.materials[i].shader = lit;
    for (int i = 0; i < model.meshCount; i++) EnsureNormals(model.meshes[i]);

    // Normalise: largest dimension 2 units, centred on the origin, resting on the ground.
    BoundingBox b = GetModelBoundingBox(model);
    Vector3 size = Vector3Subtract(b.max, b.min);
    float extent = fmaxf(size.x, fmaxf(size.y, size.z));
    modelScale = extent > 1e-6f ? 2.0f / extent : 1.0f;
    modelOffset = { -(b.min.x + b.max.x) * 0.5f * modelScale, -b.min.y * modelScale,
                    -(b.min.z + b.max.z) * 0.5f * modelScale };
    bounds = { Vector3Add(Vector3Scale(b.min, modelScale), modelOffset),
               Vector3Add(Vector3Scale(b.max, modelScale), modelOffset) };
    FrameAll();
    if (skeleton.joints.size() == 1) ResetSkeleton();   // nothing built yet: re-centre the root
}

void Viewport::ClearModel() {
    if (hasModel) UnloadModel(model);
    model = {};
    hasModel = false;
    modelName.clear();
    bounds = { { -1, 0, -1 }, { 1, 2, 1 } };
}

Matrix Viewport::ModelTransform() const {
    return MatrixMultiply(MatrixScale(modelScale, modelScale, modelScale),
                          MatrixTranslate(modelOffset.x, modelOffset.y, modelOffset.z));
}

void Viewport::FrameAll() {
    focus = Vector3Scale(Vector3Add(bounds.min, bounds.max), 0.5f);
    distance = Vector3Distance(bounds.min, bounds.max) * 1.3f + 0.5f;
}

float Viewport::JointRadius() const {
    return 0.018f * Vector3Distance(bounds.min, bounds.max);
}

// ---------------------------------------------------------------- rig editing

void Viewport::ResetSkeleton() {
    skeleton.joints.clear();
    skeleton.AddJoint(-1, Vector3Scale(Vector3Add(bounds.min, bounds.max), 0.5f), "root");
    selJoint = 0;
}

int Viewport::AddChildBoneAt(int parent, Vector3 worldPos) {
    if (parent < 0 || parent >= (int)skeleton.joints.size()) parent = 0;
    selJoint = skeleton.AddJoint(parent, worldPos, "bone_" + std::to_string(skeleton.joints.size()));
    return selJoint;
}

int Viewport::AddChildBone(int parent) {
    if (parent < 0 || parent >= (int)skeleton.joints.size()) parent = 0;
    // Continue in the direction of the parent's own bone (or the joint's +Y for the root).
    std::vector<Matrix> world = skeleton.WorldTransforms();
    Vector3 p = MatrixPosition(world[parent]);
    Vector3 dir = Vector3RotateByQuaternion({ 0, 1, 0 }, QuaternionFromMatrix(world[parent]));
    float len = JointRadius() * 8.0f;
    int grand = skeleton.joints[parent].parent;
    if (grand >= 0) {
        Vector3 bone = Vector3Subtract(p, MatrixPosition(world[grand]));
        if (Vector3Length(bone) > 1e-4f) { dir = Vector3Normalize(bone); len = Vector3Length(bone); }
    }
    return AddChildBoneAt(parent, Vector3Add(p, Vector3Scale(dir, len)));
}

void Viewport::DeleteSelectedJoint() {
    if (selJoint <= 0 || selJoint >= (int)skeleton.joints.size()) return;   // never the root
    int parent = skeleton.joints[selJoint].parent;
    skeleton.RemoveJoint(selJoint);
    selJoint = parent;
}

void Viewport::MoveJoint(int joint, Vector3 worldPos) {
    Skeleton& s = skeleton;
    // Unless the whole subtree should follow, pin the direct children in world space
    // so only the bones touching this joint change.
    std::vector<std::pair<int, Vector3>> pinned;
    if (!moveSubtree) {
        std::vector<Matrix> world = s.WorldTransforms();
        for (int i = 0; i < (int)s.joints.size(); i++)
            if (s.joints[i].parent == joint) pinned.push_back({ i, MatrixPosition(world[i]) });
    }
    s.joints[joint].offset = s.WorldToLocalOffset(s.joints[joint].parent, worldPos);
    for (auto& [child, pos] : pinned) s.joints[child].offset = s.WorldToLocalOffset(joint, pos);
}

// ---------------------------------------------------------------- picking

void Viewport::UpdateCamera() {
    float y = yaw * DEG2RAD, p = pitch * DEG2RAD;
    camera.target = focus;
    camera.position = Vector3Add(focus, { distance * cosf(p) * sinf(y), distance * sinf(p),
                                          distance * cosf(p) * cosf(y) });
    camera.up = { 0, 1, 0 };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
}

Ray Viewport::MouseRay(Vector2 local) const {
    return GetScreenToWorldRayEx(local, camera, texW, texH);
}

bool Viewport::RayModel(Ray ray, RayCollision* out) const {
    if (!hasModel || !showModel) return false;
    Matrix xf = ModelTransform();
    RayCollision best = {};
    best.distance = FLT_MAX;
    for (int i = 0; i < model.meshCount; i++) {
        RayCollision c = GetRayCollisionMesh(ray, model.meshes[i], xf);
        if (c.hit && c.distance < best.distance) best = c;
    }
    if (!best.hit) return false;
    *out = best;
    return true;
}

// Where a click-placed joint goes: inside the model (midway between the surface the ray
// enters and where it exits), else on a camera-facing plane through the parent joint.
Vector3 Viewport::PlacementPoint(Ray ray) {
    RayCollision enter;
    if (RayModel(ray, &enter)) {
        Ray inner = { Vector3Add(enter.point, Vector3Scale(ray.direction, 1e-3f)), ray.direction };
        RayCollision exit;
        if (RayModel(inner, &exit)) return Vector3Lerp(enter.point, exit.point, 0.5f);
        return enter.point;
    }
    int parent = selJoint >= 0 ? selJoint : 0;
    Vector3 planePoint = MatrixPosition(skeleton.WorldTransforms()[parent]);
    Vector3 normal = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
    float denom = Vector3DotProduct(ray.direction, normal);
    if (fabsf(denom) < 1e-6f) return planePoint;
    float t = Vector3DotProduct(Vector3Subtract(planePoint, ray.position), normal) / denom;
    return Vector3Add(ray.position, Vector3Scale(ray.direction, t));
}

int Viewport::PickJoint(Ray ray) const {
    if (!skeleton.visible) return -1;
    float best = FLT_MAX, r = JointRadius() * 1.6f;
    int hit = -1;
    std::vector<Matrix> world = skeleton.WorldTransforms();
    for (int i = 0; i < (int)world.size(); i++) {
        RayCollision c = GetRayCollisionSphere(ray, MatrixPosition(world[i]), r);
        if (c.hit && c.distance < best) { best = c.distance; hit = i; }
    }
    return hit;
}

// ---------------------------------------------------------------- gizmo

// raymath's Matrix and ImGuizmo's float[16] are both OpenGL column-major (translation in 12..14).
static void ToFloat16(const Matrix& m, float* out) {
    float16 f = MatrixToFloatV(m);
    memcpy(out, f.v, sizeof(f.v));
}

static Matrix FromFloat16(const float* f) {
    return { f[0], f[4], f[8],  f[12],
             f[1], f[5], f[9],  f[13],
             f[2], f[6], f[10], f[14],
             f[3], f[7], f[11], f[15] };
}

// Returns true while the gizmo is hovered or being dragged (so clicks don't fall through).
bool Viewport::DrawGizmo(ImVec2 pos, float w, float h) {
    if (!rigEditable || !skeleton.visible || selJoint < 0 || selJoint >= (int)skeleton.joints.size())
        return false;

    float view[16], proj[16], xf[16];
    ToFloat16(GetCameraMatrix(camera), view);
    ToFloat16(MatrixPerspective(camera.fovy * DEG2RAD, w / h, RL_CULL_DISTANCE_NEAR, RL_CULL_DISTANCE_FAR), proj);
    std::vector<Matrix> world = skeleton.WorldTransforms();
    ToFloat16(world[selJoint], xf);

    ImGuizmo::SetOrthographic(false);
    ImGuizmo::SetGizmoSizeClipSpace(0.22f);
    ImGuizmo::SetDrawlist();
    ImGuizmo::SetRect(pos.x, pos.y, w, h);
    ImGuizmo::OPERATION op = gizmoOp == GizmoOp::Translate ? ImGuizmo::TRANSLATE : ImGuizmo::ROTATE;
    if (ImGuizmo::Manipulate(view, proj, op, gizmoLocal ? ImGuizmo::LOCAL : ImGuizmo::WORLD, xf)) {
        Matrix m = FromFloat16(xf);
        if (gizmoOp == GizmoOp::Translate) {
            MoveJoint(selJoint, MatrixPosition(m));
        } else {
            // world rotation = parent * local  =>  local = parent^-1 * world
            Quaternion worldRot = QuaternionNormalize(QuaternionFromMatrix(m));
            Quaternion parentRot = QuaternionFromMatrix(skeleton.ParentWorld(selJoint, world));
            Joint& j = skeleton.joints[selJoint];
            j.rotation = ClampBallJoint(QuaternionMultiply(QuaternionInvert(parentRot), worldRot),
                                        j.swingLimit, j.twistLimit);
        }
    }
    return ImGuizmo::IsOver() || ImGuizmo::IsUsing();
}

// ---------------------------------------------------------------- input

void Viewport::HandleInput(bool hovered, bool active, Vector2 local, bool gizmoBusy) {
    ImGuiIO& io = ImGui::GetIO();

    if (hovered && io.MouseWheel != 0.0f)
        distance = Clamp(distance * powf(0.88f, io.MouseWheel), 0.2f, 200.0f);

    if (active) {
        bool orbit = ImGui::IsMouseDown(ImGuiMouseButton_Right) && !io.KeyShift;
        orbit |= ImGui::IsMouseDown(ImGuiMouseButton_Left) && io.KeyAlt;
        bool pan = ImGui::IsMouseDown(ImGuiMouseButton_Middle);
        pan |= ImGui::IsMouseDown(ImGuiMouseButton_Right) && io.KeyShift;
        if (orbit) {
            yaw -= io.MouseDelta.x * 0.4f;
            pitch = Clamp(pitch + io.MouseDelta.y * 0.4f, -89.0f, 89.0f);
        } else if (pan) {
            Vector3 fwd = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
            Vector3 right = Vector3Normalize(Vector3CrossProduct(fwd, camera.up));
            Vector3 up = Vector3CrossProduct(right, fwd);
            float k = distance * 0.0015f;
            focus = Vector3Add(focus, Vector3Scale(right, -io.MouseDelta.x * k));
            focus = Vector3Add(focus, Vector3Scale(up, io.MouseDelta.y * k));
        }
    }

    if (hovered && !io.WantTextInput) {
        if (ImGui::IsKeyPressed(ImGuiKey_F)) FrameAll();
        if (rigEditable) {
            if (ImGui::IsKeyPressed(ImGuiKey_W)) gizmoOp = GizmoOp::Translate;
            if (ImGui::IsKeyPressed(ImGuiKey_E)) gizmoOp = GizmoOp::Rotate;
            if (ImGui::IsKeyPressed(ImGuiKey_C)) AddChildBone(selJoint);
            if (ImGui::IsKeyPressed(ImGuiKey_Delete) || ImGui::IsKeyPressed(ImGuiKey_Backspace))
                DeleteSelectedJoint();
            if (ImGui::IsKeyPressed(ImGuiKey_Escape)) selJoint = -1;
        }
    }

    if (!rigEditable || gizmoBusy) return;
    if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !io.KeyAlt) {
        Ray ray = MouseRay(local);
        if (io.KeyCtrl || io.KeySuper) {
            AddChildBoneAt(selJoint, PlacementPoint(ray));   // Ctrl/Cmd+click: new child bone there
        } else {
            selJoint = PickJoint(ray);
        }
    }
}

// ---------------------------------------------------------------- UI

void Viewport::DrawToolbar() {
    if (rigEditable) {
        if (ImGui::RadioButton("Move (W)", gizmoOp == GizmoOp::Translate)) gizmoOp = GizmoOp::Translate;
        ImGui::SameLine();
        if (ImGui::RadioButton("Rotate (E)", gizmoOp == GizmoOp::Rotate)) gizmoOp = GizmoOp::Rotate;
        ImGui::SameLine();
        ImGui::Checkbox("Local axes", &gizmoLocal);
        ImGui::SameLine();
        ImGui::Checkbox("Move subtree", &moveSubtree);
        ImGui::SetItemTooltip("When moving a joint, carry its children along.\n"
                              "Off: children stay put and only the connected bones change.");
    }
    if (ImGui::Button("Frame (F)")) FrameAll();
    ImGui::SameLine();
    if (ImGui::Button("View...")) ImGui::OpenPopup("view_opts");
    if (ImGui::BeginPopup("view_opts")) {
        ImGui::Checkbox("Show model", &showModel);
        ImGui::Checkbox("Wireframe", &wireframe);
        ImGui::SliderFloat("Model opacity", &modelAlpha, 0.1f, 1.0f);
        ImGui::Checkbox("Grid", &showGrid);
        if (rigEditable) ImGui::Checkbox("Skeleton through model (x-ray)", &xraySkeleton);
        ImGui::EndPopup();
    }
    if (hasModel) {
        ImGui::SameLine();
        if (ImGui::Button("Clear model")) ClearModel();
    }
}

void Viewport::DrawPanel() {
    DrawToolbar();

    ImVec2 avail = ImGui::GetContentRegionAvail();
    int w = (int)fmaxf(avail.x, 1.0f), h = (int)fmaxf(avail.y, 1.0f);
    pixelScale = GetWindowScaleDPI().x;               // render at framebuffer resolution (Retina)
    int pw = (int)(w * pixelScale), ph = (int)(h * pixelScale);
    if (pw != texW || ph != texH) {
        if (texW > 0) UnloadRenderTexture(target);
        target = LoadRenderTexture(pw, ph);
        SetTextureFilter(target.texture, TEXTURE_FILTER_BILINEAR);
        texW = pw;
        texH = ph;
    }

    ImVec2 pos = ImGui::GetCursorScreenPos();
    screenRect = { pos.x, pos.y, (float)w, (float)h };
    ImDrawList* dl = ImGui::GetWindowDrawList();
    // render textures are stored bottom-up, so flip V
    dl->AddImage((ImTextureID)(uintptr_t)target.texture.id, pos, ImVec2(pos.x + w, pos.y + h),
                 ImVec2(0, 1), ImVec2(1, 0));

    ImU32 dim = IM_COL32(255, 255, 255, 140);
    if (!hasModel) {
        const char* msg = "Drop a 3D model here  (.obj .gltf .glb .iqm .vox .m3d)";
        ImVec2 ts = ImGui::CalcTextSize(msg);
        dl->AddText(ImVec2(pos.x + (w - ts.x) * 0.5f, pos.y + h * 0.5f - ts.y), dim, msg);
    } else {
        dl->AddText(ImVec2(pos.x + 8, pos.y + 6), dim, modelName.c_str());
    }
    const char* help = rigEditable
        ? "Click: select joint   C: add child bone   Ctrl+click: add child there   Del: delete"
        : "Reference view";
    dl->AddText(ImVec2(pos.x + 8, pos.y + h - 40), dim, help);
    dl->AddText(ImVec2(pos.x + 8, pos.y + h - 22), dim,
                "RMB/Alt+drag: orbit  MMB/Shift+RMB: pan  Wheel: zoom");

    bool gizmoBusy = DrawGizmo(pos, (float)w, (float)h);

    // The image area must not be an interactive ImGui item: ImGuizmo refuses to start a drag
    // while any item is hovered (this frame or the last). So reserve the space with a Dummy
    // and track hover / mouse capture by hand.
    ImGui::SetCursorScreenPos(pos);
    ImGui::Dummy(ImVec2((float)w, (float)h));
    bool hovered = ImGui::IsWindowHovered() &&
                   ImGui::IsMouseHoveringRect(pos, ImVec2(pos.x + w, pos.y + h));
    for (int b = 0; b < 3; b++)
        if (hovered && ImGui::IsMouseClicked(b) && !(b == 0 && gizmoBusy)) mouseCaptured = true;
    if (!ImGui::IsMouseDown(0) && !ImGui::IsMouseDown(1) && !ImGui::IsMouseDown(2)) mouseCaptured = false;

    ImVec2 mouse = ImGui::GetIO().MousePos;           // same source ImGuizmo uses
    HandleInput(hovered, mouseCaptured,
                { (mouse.x - pos.x) * pixelScale, (mouse.y - pos.y) * pixelScale }, gizmoBusy);
}

// ---------------------------------------------------------------- rendering

void Viewport::DrawSkeleton() {
    const Skeleton& s = skeleton;
    if (!s.visible) return;
    float r = JointRadius();
    std::vector<Matrix> world = s.WorldTransforms();
    Color bone = ColorAlpha(s.color, 0.85f);
    for (int i = 0; i < (int)s.joints.size(); i++) {
        int p = s.joints[i].parent;
        if (p < 0) continue;
        Vector3 a = MatrixPosition(world[p]), b = MatrixPosition(world[i]);
        DrawCylinderEx(a, b, r * 0.8f, r * 0.25f, 8, bone);
        DrawCylinderWiresEx(a, b, r * 0.8f, r * 0.25f, 8, ColorBrightness(s.color, -0.5f));
    }
    for (int i = 0; i < (int)s.joints.size(); i++) {
        Vector3 p = MatrixPosition(world[i]);
        if (i == selJoint) DrawSphereEx(p, r * 1.3f, 10, 10, WHITE);
        else if (i == 0) DrawSphereEx(p, r * 1.25f, 10, 10, ColorBrightness(s.color, -0.3f));  // root
        else DrawSphereEx(p, r, 10, 10, ColorBrightness(s.color, 0.3f));
    }

    // Selected ball joint's swing-limit cone, in the parent's frame.
    if (selJoint < 0 || selJoint >= (int)s.joints.size()) return;
    const Joint& j = s.joints[selJoint];
    if (j.swingLimit >= 179.0f) return;
    Vector3 pos = MatrixPosition(world[selJoint]);
    Quaternion parentRot = QuaternionFromMatrix(s.ParentWorld(selJoint, world));
    float len = r * 7.0f, sw = j.swingLimit * DEG2RAD;
    const int N = 32;
    Vector3 prev = {};
    for (int k = 0; k <= N; k++) {
        float a = 2 * PI * k / N;
        Vector3 d = { sinf(sw) * cosf(a), cosf(sw), sinf(sw) * sinf(a) };
        Vector3 p = Vector3Add(pos, Vector3Scale(Vector3RotateByQuaternion(d, parentRot), len));
        if (k > 0) DrawLine3D(prev, p, YELLOW);
        if (k % 4 == 0) DrawLine3D(pos, p, ColorAlpha(YELLOW, 0.5f));
        prev = p;
    }
}

void Viewport::Render(Shader lit, int viewPosLoc) {
    if (texW <= 0) return;
    UpdateCamera();

    BeginTextureMode(target);
    ClearBackground(rigEditable ? Color{ 38, 40, 46, 255 } : Color{ 34, 38, 44, 255 });
    BeginMode3D(camera);

    if (showGrid) DrawGrid(20, 0.25f);

    if (hasModel && showModel) {
        SetShaderValue(lit, viewPosLoc, &camera.position, SHADER_UNIFORM_VEC3);
        model.transform = ModelTransform();
        Color tint = ColorAlpha(WHITE, modelAlpha);
        if (modelAlpha < 0.999f) rlDisableDepthMask();
        if (wireframe) DrawModelWires(model, { 0, 0, 0 }, 1.0f, ColorAlpha(LIGHTGRAY, modelAlpha));
        else DrawModel(model, { 0, 0, 0 }, 1.0f, tint);
        rlEnableDepthMask();
    }

    if (rigEditable) {
        rlDrawRenderBatchActive();
        if (xraySkeleton) rlDisableDepthTest();
        DrawSkeleton();
        rlDrawRenderBatchActive();
        rlEnableDepthTest();
    }

    EndMode3D();
    EndTextureMode();
}
