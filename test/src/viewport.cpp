#include "viewport.hpp"
#include "rlgl.h"
#include "ImGuizmo.h"
#include <cfloat>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <map>
#include <tuple>
#include <cstdint>

Viewport::Viewport(std::string t, bool editable) : title(std::move(t)), rigEditable(editable) {
    skeleton.name = "Skeleton";
    ResetSkeleton();
    xrayModel = true;   // see-through by default, so the armature inside is visible
    if (!editable) selJoint = -1;
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
    if (rigEditable && skeleton.joints.size() == 1) ResetSkeleton();   // nothing built yet: re-centre the root
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

void Viewport::SplitSelectedBone() {
    if (selJoint <= 0 || selJoint >= (int)skeleton.joints.size()) return;   // the root has no bone
    selJoint = skeleton.SplitBone(selJoint, splitSegments);
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
    // exact up vector (derivative w.r.t. pitch), so straight top/bottom views work
    camera.up = { -sinf(p) * sinf(y), cosf(p), -sinf(p) * cosf(y) };
    camera.projection = ortho ? CAMERA_ORTHOGRAPHIC : CAMERA_PERSPECTIVE;
    // in ortho, fovy is the visible height: match what perspective shows at the focus point
    camera.fovy = ortho ? distance * 2.0f * tanf(22.5f * DEG2RAD) : 45.0f;
}

void Viewport::CameraBasis(Vector3* forward, Vector3* right, Vector3* up) const {
    *forward = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
    *right = Vector3Normalize(Vector3CrossProduct(*forward, camera.up));
    *up = Vector3CrossProduct(*right, *forward);
}

void Viewport::Orbit(float dx, float dy) {
    yaw -= dx * 0.4f;
    pitch = Clamp(pitch + dy * 0.4f, -90.0f, 90.0f);
    animating = false;
    if (autoOrtho) { ortho = false; autoOrtho = false; }   // like Blender's auto-perspective
}

static float WrapDegrees(float a) {
    a = fmodf(a + 180.0f, 360.0f);
    return (a < 0 ? a + 360.0f : a) - 180.0f;
}

void Viewport::SnapView(int axis, bool negative, bool flipIfAligned) {
    float ty = 0, tp = 0;
    if (axis == 0) ty = negative ? -90.0f : 90.0f;
    if (axis == 1) tp = negative ? -90.0f : 90.0f;
    if (axis == 2) ty = negative ? 180.0f : 0.0f;
    float curYaw = animating ? targetYaw : yaw, curPitch = animating ? targetPitch : pitch;
    bool aligned = fabsf(curPitch - tp) < 0.5f && (axis == 1 || fabsf(WrapDegrees(curYaw - ty)) < 0.5f);
    if (aligned && flipIfAligned) { SnapView(axis, !negative, false); return; }   // view from the other side

    targetYaw = curYaw + WrapDegrees(ty - curYaw);   // take the short way round
    targetPitch = tp;
    animating = true;
    if (!ortho) { ortho = true; autoOrtho = true; }
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

int Viewport::PickJoint(Ray ray, const Skeleton& s, Matrix place) const {
    if (!s.visible) return -1;
    float best = FLT_MAX, r = JointRadius() * 1.6f;
    int hit = -1;
    std::vector<Matrix> world = s.WorldTransforms();
    for (int i = 0; i < (int)world.size(); i++) {
        RayCollision c = GetRayCollisionSphere(ray, MatrixPosition(MatrixMultiply(world[i], place)), r);
        if (c.hit && c.distance < best) { best = c.distance; hit = i; }
    }
    return hit;
}

// ---------------------------------------------------------------- pose (right panel)

Skeleton Viewport::PosedSkeleton() const {
    Skeleton s = mirrorOf ? mirrorOf->skeleton : Skeleton{};
    for (Joint& j : s.joints) {
        auto it = poseRot.find(j.id);
        if (it != poseRot.end()) j.rotation = it->second;
    }
    if (!s.joints.empty()) s.joints[0].offset = Vector3Add(s.joints[0].offset, poseRootOffset);
    return s;
}

void Viewport::ResetPose() {
    poseRot.clear();
    poseRootOffset = { 0, 0, 0 };
}

// m: the gizmo's new matrix for the selected joint, already in armature space.
void Viewport::ApplyPoseGizmo(const Skeleton& posed, Matrix m) {
    std::vector<Matrix> world = posed.WorldTransforms();
    if (gizmoOp == GizmoOp::Translate) {
        if (selJoint == 0) {   // the root carries the whole armature
            poseRootOffset = Vector3Add(poseRootOffset, Vector3Subtract(MatrixPosition(m), MatrixPosition(world[0])));
            return;
        }
        // IK: bend the chain toward where the joint was dragged; bones keep their length
        Skeleton solved = posed;
        SolveIK(solved, selJoint, MatrixPosition(m), ikChain);
        for (size_t i = 0; i < solved.joints.size(); i++)
            if (!QuaternionEquals(solved.joints[i].rotation, posed.joints[i].rotation))
                poseRot[solved.joints[i].id] = solved.joints[i].rotation;
    } else {
        Quaternion worldRot = QuaternionNormalize(QuaternionFromMatrix(m));
        Quaternion parentRot = QuaternionFromMatrix(posed.ParentWorld(selJoint, world));
        const Joint& j = posed.joints[selJoint];
        poseRot[j.id] = ClampBallJoint(QuaternionMultiply(QuaternionInvert(parentRot), worldRot),
                                       j.swingLimit, j.twistLimit);
    }
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
    // Left panel edits its skeleton; the right panel poses the mirrored armature.
    bool posing = !rigEditable && mirrorOf && showMirror;
    if (!rigEditable && !posing) return false;
    Skeleton posed;
    if (posing) posed = PosedSkeleton();
    const Skeleton& sk = posing ? posed : skeleton;
    Matrix place = posing ? MirrorPlacement() : MatrixIdentity();
    if (!sk.visible || selJoint < 0 || selJoint >= (int)sk.joints.size()) return false;

    float view[16], proj[16], xf[16];
    ToFloat16(GetCameraMatrix(camera), view);
    // must match the projection BeginMode3D uses
    float top = camera.fovy * 0.5f, aspect = w / h;
    ToFloat16(ortho ? MatrixOrtho(-top * aspect, top * aspect, -top, top, RL_CULL_DISTANCE_NEAR, RL_CULL_DISTANCE_FAR)
                    : MatrixPerspective(camera.fovy * DEG2RAD, aspect, RL_CULL_DISTANCE_NEAR, RL_CULL_DISTANCE_FAR), proj);
    std::vector<Matrix> world = sk.WorldTransforms();
    ToFloat16(MatrixMultiply(world[selJoint], place), xf);

    ImGuizmo::SetOrthographic(ortho);
    ImGuizmo::SetGizmoSizeClipSpace(0.22f);
    ImGuizmo::SetDrawlist();
    ImGuizmo::SetRect(pos.x, pos.y, w, h);
    ImGuizmo::OPERATION op = gizmoOp == GizmoOp::Translate ? ImGuizmo::TRANSLATE : ImGuizmo::ROTATE;
    // ImGuizmo's drag state is global: without a per-viewport ID, dragging the gizmo in one
    // panel would also drive the gizmo drawn in the other.
    ImGuizmo::PushID(this);
    if (ImGuizmo::Manipulate(view, proj, op, gizmoLocal ? ImGuizmo::LOCAL : ImGuizmo::WORLD, xf)) {
        Matrix m = MatrixMultiply(FromFloat16(xf), MatrixInvert(place));   // back to armature space
        if (posing) {
            ApplyPoseGizmo(posed, m);
        } else if (gizmoOp == GizmoOp::Translate) {
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
    bool busy = ImGuizmo::IsOver() || ImGuizmo::IsUsing();
    ImGuizmo::PopID();
    return busy;
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
            if (io.MouseDelta.x != 0.0f || io.MouseDelta.y != 0.0f) Orbit(io.MouseDelta.x, io.MouseDelta.y);
        } else if (pan) {
            Vector3 fwd, right, up;
            CameraBasis(&fwd, &right, &up);
            float k = distance * 0.0015f;
            focus = Vector3Add(focus, Vector3Scale(right, -io.MouseDelta.x * k));
            focus = Vector3Add(focus, Vector3Scale(up, io.MouseDelta.y * k));
        }
    }

    if (hovered && !io.WantTextInput) {
        if (ImGui::IsKeyPressed(ImGuiKey_F)) FrameAll();
        // Blender-style view keys (top row or keypad); Ctrl/Cmd = view from the opposite side
        bool opposite = io.KeyCtrl || io.KeySuper;
        if (ImGui::IsKeyPressed(ImGuiKey_1) || ImGui::IsKeyPressed(ImGuiKey_Keypad1)) SnapView(2, opposite);
        if (ImGui::IsKeyPressed(ImGuiKey_3) || ImGui::IsKeyPressed(ImGuiKey_Keypad3)) SnapView(0, opposite);
        if (ImGui::IsKeyPressed(ImGuiKey_7) || ImGui::IsKeyPressed(ImGuiKey_Keypad7)) SnapView(1, opposite);
        if (ImGui::IsKeyPressed(ImGuiKey_5) || ImGui::IsKeyPressed(ImGuiKey_Keypad5)) { ortho = !ortho; autoOrtho = false; }
        if (rigEditable || mirrorOf) {
            if (ImGui::IsKeyPressed(ImGuiKey_W)) gizmoOp = GizmoOp::Translate;
            if (ImGui::IsKeyPressed(ImGuiKey_E)) gizmoOp = GizmoOp::Rotate;
            if (ImGui::IsKeyPressed(ImGuiKey_Escape)) selJoint = -1;
        }
        if (rigEditable && !(io.KeyCtrl || io.KeySuper)) {
            if (ImGui::IsKeyPressed(ImGuiKey_C)) AddChildBone(selJoint);
            if (ImGui::IsKeyPressed(ImGuiKey_Delete) || ImGui::IsKeyPressed(ImGuiKey_Backspace))
                DeleteSelectedJoint();
        }
    }

    if (gizmoBusy || !hovered || !ImGui::IsMouseClicked(ImGuiMouseButton_Left) || io.KeyAlt) return;
    Ray ray = MouseRay(local);
    if (rigEditable) {
        if (io.KeyCtrl || io.KeySuper) AddChildBoneAt(selJoint, PlacementPoint(ray));   // new child bone there
        else selJoint = PickJoint(ray, skeleton, MatrixIdentity());
    } else if (mirrorOf && showMirror) {
        selJoint = PickJoint(ray, PosedSkeleton(), MirrorPlacement());
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
    } else if (mirrorOf) {
        if (ImGui::RadioButton("Move/IK (W)", gizmoOp == GizmoOp::Translate)) gizmoOp = GizmoOp::Translate;
        ImGui::SetItemTooltip("Drag a joint: inverse kinematics bends its parents to follow.\n"
                              "Bones keep their length. Dragging the root moves the whole armature.");
        ImGui::SameLine();
        if (ImGui::RadioButton("Rotate (E)", gizmoOp == GizmoOp::Rotate)) gizmoOp = GizmoOp::Rotate;
        ImGui::SameLine();
        ImGui::SetNextItemWidth(60);
        ImGui::SliderInt("IK chain", &ikChain, 0, 10, ikChain == 0 ? "root" : "%d");
        ImGui::SetItemTooltip("How many parent joints IK may bend (root = all the way up)");
        ImGui::SameLine();
        if (ImGui::Button("Reset pose")) ResetPose();
    }
    if (ImGui::Button("Frame (F)")) FrameAll();
    ImGui::SameLine();
    ImGui::Checkbox("X-ray", &xrayModel);
    ImGui::SetItemTooltip("See-through model, so you can see the bones inside it");
    ImGui::SameLine();
    ImGui::BeginDisabled(!xrayModel);
    ImGui::SetNextItemWidth(70);
    ImGui::SliderFloat("##xray_alpha", &xrayAlpha, 0.05f, 0.9f, "%.2f");
    ImGui::SetItemTooltip("Model opacity in x-ray mode");
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Checkbox("Ortho (5)", &ortho)) autoOrtho = false;
    ImGui::SameLine();
    if (ImGui::Button("View...")) ImGui::OpenPopup("view_opts");
    if (ImGui::BeginPopup("view_opts")) {
        ImGui::Checkbox("Show model", &showModel);
        ImGui::Checkbox("Wireframe", &wireframe);
        ImGui::Checkbox("Grid", &showGrid);
        ImGui::Checkbox("Draw bones on top of the model", &xraySkeleton);
        if (mirrorOf) {
            ImGui::Checkbox("Show armature", &showMirror);
            ImGui::Checkbox("Fit armature to this model", &fitMirror);
            ImGui::SetItemTooltip("Scale and place the armature by this model's size.\n"
                                  "Off: same world coordinates as the left panel.");
        }
        if (!rigEditable) ImGui::Checkbox("Gizmo in local axes", &gizmoLocal);
        if (hasModel && ImGui::Button("Clear model")) { ClearModel(); ImGui::CloseCurrentPopup(); }
        ImGui::EndPopup();
    }
}

void Viewport::DrawPanel() {
    if (animating) {   // ease toward a snapped view
        float k = 1.0f - expf(-GetFrameTime() * 14.0f);
        yaw += (targetYaw - yaw) * k;
        pitch += (targetPitch - pitch) * k;
        if (fabsf(targetYaw - yaw) < 0.05f && fabsf(targetPitch - pitch) < 0.05f) {
            yaw = targetYaw;
            pitch = targetPitch;
            animating = false;
        }
    }
    UpdateCamera();
    if (mirrorOf && selJoint >= (int)mirrorOf->skeleton.joints.size()) selJoint = -1;
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
        ? "C: add child  Ctrl+click: add here  Del: delete  Cmd/Ctrl+Z: undo"
        : "Drag joint: IK pose (bones keep length)  E: rotate  Cmd/Ctrl+Z: undo";
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
    gizmoBusy |= DrawNavGizmo(pos, (float)w, hovered && !ImGuizmo::IsUsingAny());
    for (int b = 0; b < 3; b++)
        if (hovered && ImGui::IsMouseClicked(b) && !(b == 0 && gizmoBusy)) mouseCaptured = true;
    if (!ImGui::IsMouseDown(0) && !ImGui::IsMouseDown(1) && !ImGui::IsMouseDown(2)) mouseCaptured = false;

    ImVec2 mouse = ImGui::GetIO().MousePos;           // same source ImGuizmo uses
    HandleInput(hovered, mouseCaptured,
                { (mouse.x - pos.x) * pixelScale, (mouse.y - pos.y) * pixelScale }, gizmoBusy);
}

// Blender-style axis widget in the top-right corner: click an axis ball to look along it
// (click again for the opposite side), drag the widget to orbit. Returns true while it owns the mouse.
bool Viewport::DrawNavGizmo(ImVec2 pos, float w, bool hovered) {
    const float R = 40.0f, ball = 10.0f;
    ImVec2 c(pos.x + w - R - 22, pos.y + R + 22);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImGuiIO& io = ImGui::GetIO();

    Vector3 fwd, right, up;
    CameraBasis(&fwd, &right, &up);
    struct Ball { int axis; bool neg; ImVec2 p; float depth; };
    Ball balls[6];
    for (int i = 0; i < 6; i++) {
        Vector3 a = { 0, 0, 0 };
        (&a.x)[i / 2] = (i % 2) ? -1.0f : 1.0f;
        balls[i] = { i / 2, (i % 2) == 1,
                     ImVec2(c.x + Vector3DotProduct(a, right) * R, c.y - Vector3DotProduct(a, up) * R),
                     Vector3DotProduct(a, fwd) };   // > 0: points away from the viewer
    }
    std::sort(balls, balls + 6, [](const Ball& a, const Ball& b) { return a.depth > b.depth; });  // far first

    ImVec2 m = io.MousePos;
    bool overArea = hovered && hypotf(m.x - c.x, m.y - c.y) < R + ball + 4;
    int hot = -1;   // front-most ball under the mouse
    if (overArea && !navDragging)
        for (int i = 0; i < 6; i++)
            if (hypotf(m.x - balls[i].p.x, m.y - balls[i].p.y) < ball) hot = i;

    if (overArea || navDragging) dl->AddCircleFilled(c, R + ball + 4, IM_COL32(255, 255, 255, 28), 48);
    const ImU32 colors[3] = { IM_COL32(232, 72, 85, 255), IM_COL32(135, 200, 60, 255), IM_COL32(64, 132, 232, 255) };
    const char* names[3] = { "X", "Y", "Z" };
    for (int i = 0; i < 6; i++) {
        const Ball& b = balls[i];
        ImU32 col = colors[b.axis];
        if (!b.neg) {
            dl->AddLine(c, b.p, col, 2.5f);
            dl->AddCircleFilled(b.p, ball, col, 24);
            if (i == hot) dl->AddCircle(b.p, ball, IM_COL32_WHITE, 24, 2.0f);
            ImVec2 ts = ImGui::CalcTextSize(names[b.axis]);
            dl->AddText(ImVec2(b.p.x - ts.x * 0.5f, b.p.y - ts.y * 0.5f), IM_COL32(20, 20, 20, 255), names[b.axis]);
        } else {
            dl->AddCircleFilled(b.p, ball, (col & 0x00FFFFFF) | (i == hot ? 0xC0000000 : 0x50000000), 24);
            dl->AddCircle(b.p, ball, col, 24, 1.5f);
            if (i == hot) {
                char label[3] = { '-', names[b.axis][0], 0 };
                ImVec2 ts = ImGui::CalcTextSize(label);
                dl->AddText(ImVec2(b.p.x - ts.x * 0.5f, b.p.y - ts.y * 0.5f), IM_COL32(20, 20, 20, 255), label);
            }
        }
    }

    if (overArea && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        if (hot >= 0) SnapView(balls[hot].axis, balls[hot].neg, true);
        else navDragging = true;
    }
    if (navDragging) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            if (io.MouseDelta.x != 0.0f || io.MouseDelta.y != 0.0f) Orbit(io.MouseDelta.x, io.MouseDelta.y);
        } else {
            navDragging = false;
        }
    }
    return overArea || navDragging;
}

// ---------------------------------------------------------------- rendering

// Maps the mirrored armature from the source viewport's model space into this one's:
// the source model's base centre goes to ours, scaled by the ratio of model heights.
Matrix Viewport::MirrorPlacement() const {
    if (!mirrorOf || !fitMirror) return MatrixIdentity();
    const BoundingBox& from = mirrorOf->bounds;
    float hFrom = from.max.y - from.min.y, hTo = bounds.max.y - bounds.min.y;
    float scale = hFrom > 1e-6f ? hTo / hFrom : 1.0f;
    Vector3 a = { (from.min.x + from.max.x) * 0.5f, from.min.y, (from.min.z + from.max.z) * 0.5f };
    Vector3 b = { (bounds.min.x + bounds.max.x) * 0.5f, bounds.min.y, (bounds.min.z + bounds.max.z) * 0.5f };
    return MatrixMultiply(MatrixMultiply(MatrixTranslate(-a.x, -a.y, -a.z), MatrixScale(scale, scale, scale)),
                          MatrixTranslate(b.x, b.y, b.z));
}

void Viewport::DrawSkeleton(const Skeleton& s, int sel, Matrix place, bool showLimits) {
    if (!s.visible) return;
    float r = JointRadius();
    std::vector<Matrix> rest = s.WorldTransforms();   // armature space (no scale)
    std::vector<Matrix> world = rest;
    for (Matrix& m : world) m = MatrixMultiply(m, place);
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
        if (i == sel) DrawSphereEx(p, r * 1.3f, 10, 10, WHITE);
        else if (i == 0) DrawSphereEx(p, r * 1.25f, 10, 10, ColorBrightness(s.color, -0.3f));  // root
        else DrawSphereEx(p, r, 10, 10, ColorBrightness(s.color, 0.3f));
    }

    // Selected ball joint's swing-limit cone, in the parent's frame.
    if (!showLimits || sel < 0 || sel >= (int)s.joints.size()) return;
    const Joint& j = s.joints[sel];
    if (j.swingLimit >= 179.0f) return;
    Vector3 pos = MatrixPosition(world[sel]);
    Quaternion parentRot = QuaternionFromMatrix(s.ParentWorld(sel, rest));
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
    rlDrawRenderBatchActive();   // grid must be drawn before a see-through model blends over it

    if (hasModel && showModel) {
        SetShaderValue(lit, viewPosLoc, &camera.position, SHADER_UNIFORM_VEC3);
        model.transform = ModelTransform();
        if (wireframe) {
            DrawModelWires(model, { 0, 0, 0 }, 1.0f, ColorAlpha(LIGHTGRAY, xrayModel ? xrayAlpha : 1.0f));
        } else if (xrayModel) {
            // Depth-only pass first, so only the nearest surface gets blended
            // (otherwise the inner faces stack up into a muddy mess).
            rlColorMask(false, false, false, false);
            DrawModel(model, { 0, 0, 0 }, 1.0f, WHITE);
            rlColorMask(true, true, true, true);
            DrawModel(model, { 0, 0, 0 }, 1.0f, ColorAlpha(WHITE, xrayAlpha));
        } else {
            DrawModel(model, { 0, 0, 0 }, 1.0f, WHITE);
        }
    }

    // Our own armature, or (read-only, updated every frame) the one we mirror.
    const Skeleton* sk = rigEditable ? &skeleton : (mirrorOf && showMirror ? &mirrorOf->skeleton : nullptr);
    if (sk) {
        rlDrawRenderBatchActive();
        if (xraySkeleton) rlDisableDepthTest();
        if (rigEditable) DrawSkeleton(skeleton, selJoint, MatrixIdentity(), true);
        else DrawSkeleton(PosedSkeleton(), selJoint, MirrorPlacement(), true);
        rlDrawRenderBatchActive();
        rlEnableDepthTest();
    }

    EndMode3D();
    EndTextureMode();
}
