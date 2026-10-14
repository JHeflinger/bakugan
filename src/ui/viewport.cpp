#include "viewport.hpp"
#include "util/utils.hpp"
#include <ImGuizmo.h>
#include <algorithm>
#include <raymath.h>
#include <cstring>
#include <cstdint>
#include <rlgl.h>
#include <cfloat>
#include <cmath>
#include <tuple>
#include <map>

namespace Bakugan {
    Viewport::Viewport(std::string t, bool editable) : m_Title(std::move(t)), m_Editable(editable) {
        m_Skeleton.SetName("Skeleton");
        ResetSkeleton();
        m_XrayModel = true;
        if (!editable) m_SelectedJoint = -1;
    }

    void Viewport::Unload() {
        if (m_TexWidth > 0) UnloadRenderTexture(m_Target);
        UnloadOverlapTextures();
        ClearModel();
    }

    static void EnsureNormals(Mesh& mesh) {
        if (!mesh.vertices || mesh.vertexCount < 3) return;
        if (mesh.normals) {
            for (int i = 3; i < mesh.vertexCount * 3; i++)
                if (mesh.normals[i] != mesh.normals[i % 3]) return;
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
            rlEnableVertexArray(mesh.vaoId);
            mesh.vboId[RL_DEFAULT_SHADER_ATTRIB_LOCATION_NORMAL] = rlLoadVertexBuffer(mesh.normals, bytes, false);
            rlSetVertexAttribute(RL_DEFAULT_SHADER_ATTRIB_LOCATION_NORMAL, 3, RL_FLOAT, 0, 0, 0);
            rlEnableVertexAttribute(RL_DEFAULT_SHADER_ATTRIB_LOCATION_NORMAL);
            rlDisableVertexArray();
        }
    }

    void Viewport::SetModel(Model m, const std::string& name, Shader lit) {
        ClearModel();
        m_Model = m;
        m_HasModel = true;
        m_ModelVersion++;
        m_UserScale = 1.0f;
        m_ModelName = name;
        for (int i = 0; i < m_Model.materialCount; i++) m_Model.materials[i].shader = lit;
        for (int i = 0; i < m_Model.meshCount; i++) EnsureNormals(m_Model.meshes[i]);

        BoundingBox b = GetModelBoundingBox(m_Model);
        Vector3 size = Vector3Subtract(b.max, b.min);
        float extent = fmaxf(size.x, fmaxf(size.y, size.z));
        m_ModelScale = extent > 1e-6f ? 2.0f / extent : 1.0f;
        m_ModelOffset = { -(b.min.x + b.max.x) * 0.5f * m_ModelScale, -b.min.y * m_ModelScale,
                        -(b.min.z + b.max.z) * 0.5f * m_ModelScale };
        m_Bounds = { Vector3Add(Vector3Scale(b.min, m_ModelScale), m_ModelOffset),
                   Vector3Add(Vector3Scale(b.max, m_ModelScale), m_ModelOffset) };
        FrameAll();
        if (m_Editable && m_Skeleton.Joints().size() == 1) ResetSkeleton();   // nothing built yet: re-centre the root
    }

    void Viewport::ClearModel() {
        if (m_HasModel) UnloadModel(m_Model);
        m_Model = {};
        m_HasModel = false;
        m_ModelName.clear();
        m_Bounds = { { -1, 0, -1 }, { 1, 2, 1 } };
        m_ModelVersion++;
        m_WorldMesh.reset();
    }

    Matrix Viewport::ModelTransform() const {
        Matrix fit = MatrixMultiply(MatrixScale(m_ModelScale, m_ModelScale, m_ModelScale),
                                    MatrixTranslate(m_ModelOffset.x, m_ModelOffset.y, m_ModelOffset.z));
        if (m_UserScale == 1.0f) return fit;
        Vector3 c = Vector3Scale(Vector3Add(m_Bounds.min, m_Bounds.max), 0.5f);
        Matrix about = MatrixMultiply(MatrixMultiply(MatrixTranslate(-c.x, -c.y, -c.z), MatrixScale(m_UserScale, m_UserScale, m_UserScale)),
                                      MatrixTranslate(c.x, c.y, c.z));
        return MatrixMultiply(fit, about);
    }

    void Viewport::FrameAll() {
        m_Focus = Vector3Scale(Vector3Add(m_Bounds.min, m_Bounds.max), 0.5f);
        m_Distance = Vector3Distance(m_Bounds.min, m_Bounds.max) * 1.3f + 0.5f;
    }

    float Viewport::JointRadius() const {
        return 0.018f * Vector3Distance(m_Bounds.min, m_Bounds.max);
    }

    void Viewport::ResetSkeleton() {
        m_Skeleton.JointsRef().clear();
        m_Skeleton.AddJoint(-1, Vector3Scale(Vector3Add(m_Bounds.min, m_Bounds.max), 0.5f), "root");
        m_SelectedJoint = 0;
    }

    int Viewport::AddChildBoneAt(int parent, Vector3 worldPos) {
        if (parent < 0 || parent >= (int)m_Skeleton.Joints().size()) parent = 0;
        m_SelectedJoint = m_Skeleton.AddJoint(parent, worldPos, "bone_" + std::to_string(m_Skeleton.Joints().size()));
        return m_SelectedJoint;
    }

    int Viewport::AddChildBone(int parent) {
        if (parent < 0 || parent >= (int)m_Skeleton.Joints().size()) parent = 0;
        std::vector<Matrix> world = m_Skeleton.WorldTransforms();
        Vector3 p = Utils::MatrixPosition(world[parent]);
        Vector3 dir = Vector3RotateByQuaternion({ 0, 1, 0 }, QuaternionFromMatrix(world[parent]));
        float len = JointRadius() * 8.0f;
        int grand = m_Skeleton.Joints()[parent].parent;
        if (grand >= 0) {
            Vector3 bone = Vector3Subtract(p, Utils::MatrixPosition(world[grand]));
            if (Vector3Length(bone) > 1e-4f) { dir = Vector3Normalize(bone); len = Vector3Length(bone); }
        }
        return AddChildBoneAt(parent, Vector3Add(p, Vector3Scale(dir, len)));
    }

    void Viewport::DeleteSelectedJoint() {
        if (m_SelectedJoint <= 0 || m_SelectedJoint >= (int)m_Skeleton.Joints().size()) return;   // never the root
        int parent = m_Skeleton.Joints()[m_SelectedJoint].parent;
        m_Skeleton.RemoveJoint(m_SelectedJoint);
        m_SelectedJoint = parent;
    }

    void Viewport::SplitSelectedBone() {
        if (m_SelectedJoint <= 0 || m_SelectedJoint >= (int)m_Skeleton.Joints().size()) return;   // the root has no bone
        m_SelectedJoint = m_Skeleton.SplitBone(m_SelectedJoint, m_SplitSegments);
    }

    void Viewport::MoveJoint(int joint, Vector3 worldPos) {
        Skeleton& s = m_Skeleton;
        std::vector<std::pair<int, Vector3>> pinned;
        if (!m_MoveSubtree) {
            std::vector<Matrix> world = s.WorldTransforms();
            for (int i = 0; i < (int)s.Joints().size(); i++)
                if (s.Joints()[i].parent == joint) pinned.push_back({ i, Utils::MatrixPosition(world[i]) });
        }
        s.JointsRef()[joint].offset = s.WorldToLocalOffset(s.Joints()[joint].parent, worldPos);
        for (auto& [child, pos] : pinned) s.JointsRef()[child].offset = s.WorldToLocalOffset(joint, pos);
    }

    void Viewport::UpdateCamera() {
        float y = m_Yaw * DEG2RAD, p = m_Pitch * DEG2RAD;
        m_Camera.target = m_Focus;
        m_Camera.position = Vector3Add(m_Focus, { m_Distance * cosf(p) * sinf(y), m_Distance * sinf(p),
                                              m_Distance * cosf(p) * cosf(y) });
        m_Camera.up = { -sinf(p) * sinf(y), cosf(p), -sinf(p) * cosf(y) };
        m_Camera.projection = m_Ortho ? CAMERA_ORTHOGRAPHIC : CAMERA_PERSPECTIVE;
        m_Camera.fovy = m_Ortho ? m_Distance * 2.0f * tanf(22.5f * DEG2RAD) : 45.0f;
    }

    void Viewport::CameraBasis(Vector3* forward, Vector3* right, Vector3* up) const {
        *forward = Vector3Normalize(Vector3Subtract(m_Camera.target, m_Camera.position));
        *right = Vector3Normalize(Vector3CrossProduct(*forward, m_Camera.up));
        *up = Vector3CrossProduct(*right, *forward);
    }

    void Viewport::Orbit(float dx, float dy) {
        m_Yaw -= dx * 0.4f;
        m_Pitch = Clamp(m_Pitch + dy * 0.4f, -90.0f, 90.0f);
        m_Animating = false;
        if (m_AutoOrtho) { m_Ortho = false; m_AutoOrtho = false; }
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
        float curYaw = m_Animating ? m_TargetYaw : m_Yaw, curPitch = m_Animating ? m_TargetPitch : m_Pitch;
        bool aligned = fabsf(curPitch - tp) < 0.5f && (axis == 1 || fabsf(WrapDegrees(curYaw - ty)) < 0.5f);
        if (aligned && flipIfAligned) { SnapView(axis, !negative, false); return; }   // view from the other side
        m_TargetYaw = curYaw + WrapDegrees(ty - curYaw);
        m_TargetPitch = tp;
        m_Animating = true;
        if (!m_Ortho) { m_Ortho = true; m_AutoOrtho = true; }
    }

    Ray Viewport::MouseRay(Vector2 local) const {
        return GetScreenToWorldRayEx(local, m_Camera, m_TexWidth, m_TexHeight);
    }

    bool Viewport::RayModel(Ray ray, RayCollision* out) const {
        if (!m_HasModel || !m_ShowModel) return false;
        Matrix xf = ModelTransform();
        RayCollision best = {};
        best.distance = FLT_MAX;
        for (int i = 0; i < m_Model.meshCount; i++) {
            RayCollision c = GetRayCollisionMesh(ray, m_Model.meshes[i], xf);
            if (c.hit && c.distance < best.distance) best = c;
        }
        if (!best.hit) return false;
        *out = best;
        return true;
    }

    Vector3 Viewport::PlacementPoint(Ray ray) {
        RayCollision enter;
        if (RayModel(ray, &enter)) {
            Ray inner = { Vector3Add(enter.point, Vector3Scale(ray.direction, 1e-3f)), ray.direction };
            RayCollision exit;
            if (RayModel(inner, &exit)) return Vector3Lerp(enter.point, exit.point, 0.5f);
            return enter.point;
        }
        int parent = m_SelectedJoint >= 0 ? m_SelectedJoint : 0;
        Vector3 planePoint = Utils::MatrixPosition(m_Skeleton.WorldTransforms()[parent]);
        Vector3 normal = Vector3Normalize(Vector3Subtract(m_Camera.target, m_Camera.position));
        float denom = Vector3DotProduct(ray.direction, normal);
        if (fabsf(denom) < 1e-6f) return planePoint;
        float t = Vector3DotProduct(Vector3Subtract(planePoint, ray.position), normal) / denom;
        return Vector3Add(ray.position, Vector3Scale(ray.direction, t));
    }

    int Viewport::PickJoint(Ray ray, const Skeleton& s, Matrix place) const {
        if (!s.IsVisible()) return -1;
        float best = FLT_MAX, r = JointRadius() * 1.6f;
        int hit = -1;
        std::vector<Matrix> world = s.WorldTransforms();
        for (int i = 0; i < (int)world.size(); i++) {
            RayCollision c = GetRayCollisionSphere(ray, Utils::MatrixPosition(MatrixMultiply(world[i], place)), r);
            if (c.hit && c.distance < best) { best = c.distance; hit = i; }
        }
        return hit;
    }

    Skeleton Viewport::PosedSkeleton() const {
        Skeleton s = m_Mirror ? m_Mirror->GetSkeleton() : Skeleton{};
        for (Joint& j : s.JointsRef()) {
            auto it = m_PoseRotations.find(j.id);
            if (it != m_PoseRotations.end()) j.rotation = it->second;
        }
        if (!s.Joints().empty()) s.JointsRef()[0].offset = Vector3Add(s.Joints()[0].offset, m_PoseRootOffset);
        return s;
    }

    void Viewport::ResetPose() {
        m_PoseRotations.clear();
        m_PoseRootOffset = { 0, 0, 0 };
    }

    void Viewport::ApplyPoseGizmo(const Skeleton& posed, Matrix m) {
        std::vector<Matrix> world = posed.WorldTransforms();
        if (m_GizmoOperation == GizmoOp::Translate) {
            if (m_SelectedJoint == 0) {
                m_PoseRootOffset = Vector3Add(m_PoseRootOffset, Vector3Subtract(Utils::MatrixPosition(m), Utils::MatrixPosition(world[0])));
                return;
            }
            Skeleton solved = posed;
            Utils::SolveIK(solved, m_SelectedJoint, Utils::MatrixPosition(m), m_IKChain);
            for (size_t i = 0; i < solved.Joints().size(); i++)
                if (!QuaternionEquals(solved.Joints()[i].rotation, posed.Joints()[i].rotation))
                    m_PoseRotations[solved.Joints()[i].id] = solved.Joints()[i].rotation;
        } else {
            Quaternion worldRot = QuaternionNormalize(QuaternionFromMatrix(m));
            Quaternion parentRot = QuaternionFromMatrix(posed.ParentWorld(m_SelectedJoint, world));
            const Joint& j = posed.Joints()[m_SelectedJoint];
            m_PoseRotations[j.id] = Utils::ClampBallJoint(QuaternionMultiply(QuaternionInvert(parentRot), worldRot),
                                           j.swingLimit, j.twistLimit);
        }
    }

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

    bool Viewport::DrawGizmo(ImVec2 pos, float w, float h) {
        bool posing = !m_Editable && m_Mirror && m_ShowMirror;
        if (!m_Editable && !posing) return false;
        Skeleton posed;
        if (posing) posed = PosedSkeleton();
        const Skeleton& sk = posing ? posed : m_Skeleton;
        Matrix place = posing ? MirrorPlacement() : MatrixIdentity();
        if (!sk.IsVisible() || m_SelectedJoint < 0 || m_SelectedJoint >= (int)sk.Joints().size()) return false;
        float view[16], proj[16], xf[16];
        ToFloat16(GetCameraMatrix(m_Camera), view);
        float top = m_Camera.fovy * 0.5f, aspect = w / h;
        ToFloat16(m_Ortho ? MatrixOrtho(-top * aspect, top * aspect, -top, top, RL_CULL_DISTANCE_NEAR, RL_CULL_DISTANCE_FAR)
                        : MatrixPerspective(m_Camera.fovy * DEG2RAD, aspect, RL_CULL_DISTANCE_NEAR, RL_CULL_DISTANCE_FAR), proj);
        std::vector<Matrix> world = sk.WorldTransforms();
        ToFloat16(MatrixMultiply(world[m_SelectedJoint], place), xf);
        ImGuizmo::SetOrthographic(m_Ortho);
        ImGuizmo::SetGizmoSizeClipSpace(0.22f);
        ImGuizmo::SetDrawlist();
        ImGuizmo::SetRect(pos.x, pos.y, w, h);
        ImGuizmo::OPERATION op = m_GizmoOperation == GizmoOp::Translate ? ImGuizmo::TRANSLATE : ImGuizmo::ROTATE;
        ImGuizmo::PushID(this);
        if (ImGuizmo::Manipulate(view, proj, op, m_LocalGizmo ? ImGuizmo::LOCAL : ImGuizmo::WORLD, xf)) {
            Matrix m = MatrixMultiply(FromFloat16(xf), MatrixInvert(place));   // back to armature space
            if (posing) {
                ApplyPoseGizmo(posed, m);
            } else if (m_GizmoOperation == GizmoOp::Translate) {
                MoveJoint(m_SelectedJoint, Utils::MatrixPosition(m));
            } else {
                Quaternion worldRot = QuaternionNormalize(QuaternionFromMatrix(m));
                Quaternion parentRot = QuaternionFromMatrix(m_Skeleton.ParentWorld(m_SelectedJoint, world));
                Joint& j = m_Skeleton.JointsRef()[m_SelectedJoint];
                j.rotation = Utils::ClampBallJoint(QuaternionMultiply(QuaternionInvert(parentRot), worldRot),
                                            j.swingLimit, j.twistLimit);
            }
        }
        bool busy = ImGuizmo::IsOver() || ImGuizmo::IsUsing();
        ImGuizmo::PopID();
        return busy;
    }

    void Viewport::HandleInput(bool hovered, bool active, Vector2 local, bool gizmoBusy) {
        ImGuiIO& io = ImGui::GetIO();
        if (hovered && io.MouseWheel != 0.0f)
            m_Distance = Clamp(m_Distance * powf(0.88f, io.MouseWheel), 0.2f, 200.0f);
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
                float k = m_Distance * 0.0015f;
                m_Focus = Vector3Add(m_Focus, Vector3Scale(right, -io.MouseDelta.x * k));
                m_Focus = Vector3Add(m_Focus, Vector3Scale(up, io.MouseDelta.y * k));
            }
        }
        if (hovered && !io.WantTextInput) {
            if (ImGui::IsKeyPressed(ImGuiKey_F)) FrameAll();
            bool opposite = io.KeyCtrl || io.KeySuper;
            if (ImGui::IsKeyPressed(ImGuiKey_1) || ImGui::IsKeyPressed(ImGuiKey_Keypad1)) SnapView(2, opposite);
            if (ImGui::IsKeyPressed(ImGuiKey_3) || ImGui::IsKeyPressed(ImGuiKey_Keypad3)) SnapView(0, opposite);
            if (ImGui::IsKeyPressed(ImGuiKey_7) || ImGui::IsKeyPressed(ImGuiKey_Keypad7)) SnapView(1, opposite);
            if (ImGui::IsKeyPressed(ImGuiKey_5) || ImGui::IsKeyPressed(ImGuiKey_Keypad5)) { m_Ortho = !m_Ortho; m_AutoOrtho = false; }
            if (m_Editable || m_Mirror) {
                if (ImGui::IsKeyPressed(ImGuiKey_W)) m_GizmoOperation = GizmoOp::Translate;
                if (ImGui::IsKeyPressed(ImGuiKey_E)) m_GizmoOperation = GizmoOp::Rotate;
                if (ImGui::IsKeyPressed(ImGuiKey_Escape)) m_SelectedJoint = -1;
            }
            if (m_Editable && !(io.KeyCtrl || io.KeySuper)) {
                if (ImGui::IsKeyPressed(ImGuiKey_C)) AddChildBone(m_SelectedJoint);
                if (ImGui::IsKeyPressed(ImGuiKey_Delete) || ImGui::IsKeyPressed(ImGuiKey_Backspace))
                    DeleteSelectedJoint();
            }
        }
        if (gizmoBusy || !hovered || !ImGui::IsMouseClicked(ImGuiMouseButton_Left) || io.KeyAlt) return;
        Ray ray = MouseRay(local);
        if (m_Editable) {
            if (io.KeyCtrl || io.KeySuper) AddChildBoneAt(m_SelectedJoint, PlacementPoint(ray));   // new child bone there
            else m_SelectedJoint = PickJoint(ray, m_Skeleton, MatrixIdentity());
        } else if (m_Mirror && m_ShowMirror) {
            m_SelectedJoint = PickJoint(ray, PosedSkeleton(), MirrorPlacement());
        }
    }

    void Viewport::DrawToolbar() {
        if (m_Editable) {
            if (ImGui::RadioButton("Move (W)", m_GizmoOperation == GizmoOp::Translate)) m_GizmoOperation = GizmoOp::Translate;
            ImGui::SameLine();
            if (ImGui::RadioButton("Rotate (E)", m_GizmoOperation == GizmoOp::Rotate)) m_GizmoOperation = GizmoOp::Rotate;
            ImGui::SameLine();
            ImGui::Checkbox("Local axes", &m_LocalGizmo);
            ImGui::SameLine();
            ImGui::Checkbox("Move subtree", &m_MoveSubtree);
            ImGui::SetItemTooltip("When moving a joint, carry its children along.\n"
                                  "Off: children stay put and only the connected bones change.");
        } else if (m_Mirror) {
            if (ImGui::RadioButton("Move/IK (W)", m_GizmoOperation == GizmoOp::Translate)) m_GizmoOperation = GizmoOp::Translate;
            ImGui::SetItemTooltip("Drag a joint: inverse kinematics bends its parents to follow.\n"
                                  "Bones keep their length. Dragging the root moves the whole armature.");
            ImGui::SameLine();
            if (ImGui::RadioButton("Rotate (E)", m_GizmoOperation == GizmoOp::Rotate)) m_GizmoOperation = GizmoOp::Rotate;
            ImGui::SameLine();
            ImGui::SetNextItemWidth(60);
            ImGui::SliderInt("IK chain", &m_IKChain, 0, 10, m_IKChain == 0 ? "root" : "%d");
            ImGui::SetItemTooltip("How many parent Joints() IK may bend (root = all the way up)");
            ImGui::SameLine();
            if (ImGui::Button("Reset pose")) ResetPose();
        }
        if (ImGui::Button("Frame (F)")) FrameAll();
        ImGui::SameLine();
        ImGui::Checkbox("X-ray", &m_XrayModel);
        ImGui::SetItemTooltip("See-through model, so you can see the bones inside it");
        ImGui::SameLine();
        ImGui::BeginDisabled(!m_XrayModel);
        ImGui::SetNextItemWidth(70);
        ImGui::SliderFloat("##xray_alpha", &m_XrayAlpha, 0.05f, 0.9f, "%.2f");
        ImGui::SetItemTooltip("Model opacity in x-ray mode");
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Checkbox("Ortho (5)", &m_Ortho)) m_AutoOrtho = false;
        ImGui::SameLine();
        if (ImGui::Button("View...")) ImGui::OpenPopup("view_opts");
        if (ImGui::BeginPopup("view_opts")) {
            ImGui::Checkbox("Show model", &m_ShowModel);
            ImGui::Checkbox("m_Wireframe", &m_Wireframe);
            ImGui::Checkbox("Grid", &m_ShowGrid);
            ImGui::Checkbox("Draw bones on top of the model", &m_XraySkeleton);
            if (m_Mirror) {
                ImGui::Checkbox("Show armature", &m_ShowMirror);
                ImGui::Checkbox("Fit armature to this model", &m_FitMirror);
                ImGui::SetItemTooltip("Scale and place the armature by this model's size.\n"
                                      "Off: same world coordinates as the left panel.");
            }
            if (!m_Editable) ImGui::Checkbox("Gizmo in local axes", &m_LocalGizmo);
            if (m_HasModel && ImGui::Button("Clear model")) { ClearModel(); ImGui::CloseCurrentPopup(); }
            ImGui::EndPopup();
        }
    }

    void Viewport::DrawPanel() {
        if (m_Animating) {
            float k = 1.0f - expf(-GetFrameTime() * 14.0f);
            m_Yaw += (m_TargetYaw - m_Yaw) * k;
            m_Pitch += (m_TargetPitch - m_Pitch) * k;
            if (fabsf(m_TargetYaw - m_Yaw) < 0.05f && fabsf(m_TargetPitch - m_Pitch) < 0.05f) {
                m_Yaw = m_TargetYaw;
                m_Pitch = m_TargetPitch;
                m_Animating = false;
            }
        }
        UpdateCamera();
        if (m_Mirror && m_SelectedJoint >= (int)m_Mirror->m_Skeleton.Joints().size()) m_SelectedJoint = -1;
        DrawToolbar();
        ImVec2 avail = ImGui::GetContentRegionAvail();
        int w = (int)fmaxf(avail.x, 1.0f), h = (int)fmaxf(avail.y, 1.0f);
        m_PixelScale = GetWindowScaleDPI().x;
        int pw = (int)(w * m_PixelScale), ph = (int)(h * m_PixelScale);
        if (pw != m_TexWidth || ph != m_TexHeight) {
            if (m_TexWidth > 0) UnloadRenderTexture(m_Target);
            m_Target = LoadRenderTexture(pw, ph);
            SetTextureFilter(m_Target.texture, TEXTURE_FILTER_BILINEAR);
            m_TexWidth = pw;
            m_TexHeight = ph;
        }
        ImVec2 pos = ImGui::GetCursorScreenPos();
        m_ScreenRectangle = { pos.x, pos.y, (float)w, (float)h };
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddImage((ImTextureID)(uintptr_t)m_Target.texture.id, pos, ImVec2(pos.x + w, pos.y + h),
                     ImVec2(0, 1), ImVec2(1, 0));
        ImU32 dim = IM_COL32(255, 255, 255, 140);
        if (!m_HasModel) {
            const char* msg = "Drop a 3D model here  (.obj .gltf .glb .iqm .vox .m3d)";
            ImVec2 ts = ImGui::CalcTextSize(msg);
            dl->AddText(ImVec2(pos.x + (w - ts.x) * 0.5f, pos.y + h * 0.5f - ts.y), dim, msg);
        } else {
            dl->AddText(ImVec2(pos.x + 8, pos.y + 6), dim, m_ModelName.c_str());
        }
        const char* help = m_Editable
            ? "C: add child  Ctrl+click: add here  Del: delete  Cmd/Ctrl+Z: undo"
            : "Drag joint: IK pose (bones keep length)  E: rotate  Cmd/Ctrl+Z: undo";
        dl->AddText(ImVec2(pos.x + 8, pos.y + h - 40), dim, help);
        dl->AddText(ImVec2(pos.x + 8, pos.y + h - 22), dim,
                    "RMB/Alt+drag: orbit  MMB/Shift+RMB: pan  Wheel: zoom");

        bool gizmoBusy = DrawGizmo(pos, (float)w, (float)h);
        ImGui::SetCursorScreenPos(pos);
        ImGui::Dummy(ImVec2((float)w, (float)h));
        bool hovered = ImGui::IsWindowHovered() &&
                       ImGui::IsMouseHoveringRect(pos, ImVec2(pos.x + w, pos.y + h));
        gizmoBusy |= DrawNavGizmo(pos, (float)w, hovered && !ImGuizmo::IsUsingAny());
        for (int b = 0; b < 3; b++)
            if (hovered && ImGui::IsMouseClicked(b) && !(b == 0 && gizmoBusy)) m_MouseCaptured = true;
        if (!ImGui::IsMouseDown(0) && !ImGui::IsMouseDown(1) && !ImGui::IsMouseDown(2)) m_MouseCaptured = false;
        ImVec2 mouse = ImGui::GetIO().MousePos;
        HandleInput(hovered, m_MouseCaptured,
                    { (mouse.x - pos.x) * m_PixelScale, (mouse.y - pos.y) * m_PixelScale }, gizmoBusy);
    }

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
                         Vector3DotProduct(a, fwd) };
        }
        std::sort(balls, balls + 6, [](const Ball& a, const Ball& b) { return a.depth > b.depth; });  // far first
        ImVec2 m = io.MousePos;
        bool overArea = hovered && hypotf(m.x - c.x, m.y - c.y) < R + ball + 4;
        int hot = -1;
        if (overArea && !m_NavDragging)
            for (int i = 0; i < 6; i++)
                if (hypotf(m.x - balls[i].p.x, m.y - balls[i].p.y) < ball) hot = i;
        if (overArea || m_NavDragging) dl->AddCircleFilled(c, R + ball + 4, IM_COL32(255, 255, 255, 28), 48);
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
            else m_NavDragging = true;
        }
        if (m_NavDragging) {
            if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                if (io.MouseDelta.x != 0.0f || io.MouseDelta.y != 0.0f) Orbit(io.MouseDelta.x, io.MouseDelta.y);
            } else {
                m_NavDragging = false;
            }
        }
        return overArea || m_NavDragging;
    }

    Matrix Viewport::MirrorPlacement() const {
        if (!m_Mirror || !m_FitMirror) return MatrixIdentity();
        const BoundingBox& from = m_Mirror->m_Bounds;
        float hFrom = from.max.y - from.min.y, hTo = m_Bounds.max.y - m_Bounds.min.y;
        float scale = hFrom > 1e-6f ? hTo / hFrom : 1.0f;
        Vector3 a = { (from.min.x + from.max.x) * 0.5f, from.min.y, (from.min.z + from.max.z) * 0.5f };
        Vector3 b = { (m_Bounds.min.x + m_Bounds.max.x) * 0.5f, m_Bounds.min.y, (m_Bounds.min.z + m_Bounds.max.z) * 0.5f };
        return MatrixMultiply(MatrixMultiply(MatrixTranslate(-a.x, -a.y, -a.z), MatrixScale(scale, scale, scale)),
                              MatrixTranslate(b.x, b.y, b.z));
    }

    void Viewport::DrawSkeleton(const Skeleton& s, int sel, Matrix place, bool showLimits) {
        if (!s.IsVisible()) return;
        float r = JointRadius();
        std::vector<Matrix> rest = s.WorldTransforms();
        std::vector<Matrix> world = rest;
        for (Matrix& m : world) m = MatrixMultiply(m, place);
        Color bone = ColorAlpha(s.GetColor(), 0.85f);
        for (int i = 0; i < (int)s.Joints().size(); i++) {
            int p = s.Joints()[i].parent;
            if (p < 0) continue;
            Vector3 a = Utils::MatrixPosition(world[p]), b = Utils::MatrixPosition(world[i]);
            DrawCylinderEx(a, b, r * 0.8f, r * 0.25f, 8, bone);
            DrawCylinderWiresEx(a, b, r * 0.8f, r * 0.25f, 8, ColorBrightness(s.GetColor(), -0.5f));
        }
        for (int i = 0; i < (int)s.Joints().size(); i++) {
            Vector3 p = Utils::MatrixPosition(world[i]);
            if (i == sel) DrawSphereEx(p, r * 1.3f, 10, 10, WHITE);
            else if (i == 0) DrawSphereEx(p, r * 1.25f, 10, 10, ColorBrightness(s.GetColor(), -0.3f));  // root
            else DrawSphereEx(p, r, 10, 10, ColorBrightness(s.GetColor(), 0.3f));
        }

        if (!showLimits || sel < 0 || sel >= (int)s.Joints().size()) return;
        const Joint& j = s.Joints()[sel];
        if (j.swingLimit >= 179.0f) return;
        Vector3 pos = Utils::MatrixPosition(world[sel]);
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
        if (m_TexWidth <= 0) return;
        UpdateCamera();
        if (!m_Editable) UpdateOverlap();
        else UpdateFollowOverlap();
        BeginTextureMode(m_Target);
        ClearBackground(m_Editable ? Color{ 38, 40, 46, 255 } : Color{ 34, 38, 44, 255 });
        BeginMode3D(m_Camera);
        if (m_ShowGrid) DrawGrid(20, 0.25f);
        rlDrawRenderBatchActive();
        if (!m_Editable && m_ShowOverlap) DrawOverlapVolume();
        else if (m_Editable && FollowingOverlap()) DrawFollowOverlap();
        if (m_HasModel && m_ShowModel) {
            SetShaderValue(lit, viewPosLoc, &m_Camera.position, SHADER_UNIFORM_VEC3);
            m_Model.transform = ModelTransform();
            float shell = ((!m_Editable && m_ShowOverlap) || (m_Editable && FollowingOverlap())) ? fminf(m_XrayAlpha, 0.1f) : m_XrayAlpha;
            if (m_Wireframe) {
                DrawModelWires(m_Model, { 0, 0, 0 }, 1.0f, ColorAlpha(LIGHTGRAY, m_XrayModel ? shell : 1.0f));
            } else if (m_XrayModel) {
                rlColorMask(false, false, false, false);
                DrawModel(m_Model, { 0, 0, 0 }, 1.0f, WHITE);
                rlColorMask(true, true, true, true);
                DrawModel(m_Model, { 0, 0, 0 }, 1.0f, ColorAlpha(WHITE, shell));
            } else {
                DrawModel(m_Model, { 0, 0, 0 }, 1.0f, WHITE);
            }
        }
        const Skeleton* sk = m_Editable ? &m_Skeleton : (m_Mirror && m_ShowMirror ? &m_Mirror->m_Skeleton : nullptr);
        if (sk) {
            rlDrawRenderBatchActive();
            if (m_XraySkeleton) rlDisableDepthTest();
            if (m_Editable) DrawSkeleton(m_Skeleton, m_SelectedJoint, MatrixIdentity(), true);
            else DrawSkeleton(PosedSkeleton(), m_SelectedJoint, MirrorPlacement(), true);
            rlDrawRenderBatchActive();
            rlEnableDepthTest();
        }
        EndMode3D();
        EndTextureMode();
    }

    std::shared_ptr<const MeshData> Viewport::WorldMesh() const {
        if (!m_HasModel) return nullptr;
        if (m_WorldMesh && m_WorldMeshScale == m_UserScale) return m_WorldMesh;
        auto data = std::make_shared<MeshData>();
        data->id = MeshData::NextId();
        Matrix xf = ModelTransform();
        Vector3 lo = { FLT_MAX, FLT_MAX, FLT_MAX }, hi = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
        for (int m = 0; m < m_Model.meshCount; m++) {
            const Mesh& mesh = m_Model.meshes[m];
            if (!mesh.vertices || mesh.vertexCount < 3) continue;
            unsigned base = (unsigned)data->vertices.size();
            for (int i = 0; i < mesh.vertexCount; i++) {
                Vector3 p = Vector3Transform({ mesh.vertices[i * 3], mesh.vertices[i * 3 + 1], mesh.vertices[i * 3 + 2] }, xf);
                data->vertices.push_back(p);
                lo = Vector3Min(lo, p);
                hi = Vector3Max(hi, p);
            }
            int tris = mesh.indices ? mesh.triangleCount : mesh.vertexCount / 3;
            for (int t = 0; t < tris; t++)
                for (int k = 0; k < 3; k++)
                    data->triangles.push_back(base + (mesh.indices ? mesh.indices[t * 3 + k] : (unsigned)(t * 3 + k)));
        }
        if (data->triangles.empty()) return nullptr;
        data->box = { lo, hi };
        m_WorldMesh = data;
        m_WorldMeshScale = m_UserScale;
        return m_WorldMesh;
    }

    static void HashBytes(uint64_t& h, const void* data, size_t size) {
        const unsigned char* p = (const unsigned char*)data;
        for (size_t i = 0; i < size; i++) { h ^= p[i]; h *= 1099511628211ull; }
    }

    template <class T> static void HashValue(uint64_t& h, const T& v) { HashBytes(h, &v, sizeof(T)); }

    static void HashSkeleton(uint64_t& h, const Skeleton& s) {
        for (const Joint& j : s.Joints()) {
            HashValue(h, j.id);
            HashValue(h, j.parent);
            HashValue(h, j.offset);
            HashValue(h, j.rotation);
        }
        HashValue(h, s.Joints().size());
    }

    uint64_t Viewport::OverlapSignature() const {
        uint64_t h = 1469598103934665603ull;
        HashValue(h, m_ModelVersion);
        HashValue(h, m_UserScale);
        HashValue(h, m_Mirror->m_ModelVersion);
        HashValue(h, m_OverlapResolution);
        HashValue(h, m_OverlapErosion);
        HashSkeleton(h, m_Mirror->m_Skeleton);
        HashSkeleton(h, PosedSkeleton());
        HashValue(h, MirrorPlacement());
        return h ? h : 1;
    }

    void Viewport::UpdateOverlap() {
        OverlapResult finished;
        if (m_Overlap.Poll(&finished) && m_ShowOverlap) {
            m_OverlapResult = std::move(finished);
            m_OverlapGeneration++;
            m_OverlapDirty = true;
        }
        bool ready = m_ShowOverlap && m_Mirror && m_HasModel && m_Mirror->m_HasModel;
        if (!ready) {
            if (!m_OverlapSlices[0].empty()) UnloadOverlapTextures();
            m_OverlapSignature = 0;
            m_OverlapDirty = false;
            return;
        }
        uint64_t sig = OverlapSignature();
        if (sig != m_OverlapSignature) {
            OverlapRequest rq;
            rq.source = m_Mirror->WorldMesh();
            rq.target = WorldMesh();
            Skeleton posed = PosedSkeleton();
            rq.restWorld = m_Mirror->m_Skeleton.WorldTransforms();
            rq.posedWorld = posed.WorldTransforms();
            for (const Joint& j : m_Mirror->m_Skeleton.Joints()) rq.parent.push_back(j.parent);
            rq.place = MirrorPlacement();
            rq.params.resolution = m_OverlapResolution;
            rq.params.erosion = m_OverlapErosion;
            rq.signature = sig;
            m_Overlap.Request(std::move(rq));
            m_OverlapSignature = sig;
        }
        if (m_OverlapDirty) BuildOverlapTextures();
    }

    static void UnloadStack(std::vector<Texture2D> (&stack)[3]) {
        for (auto& slices : stack) {
            for (Texture2D& t : slices) UnloadTexture(t);
            slices.clear();
        }
    }

    void Viewport::UnloadOverlapTextures() {
        UnloadStack(m_OverlapSlices);
        UnloadStack(m_FollowSlices);
        m_FollowGeneration = 0;
    }

    static void BuildStack(std::vector<Texture2D> (&stack)[3], const std::vector<uint8_t>& cells, const int dim[3], bool onlyOverlap, float opacity) {
        UnloadStack(stack);
        if (cells.empty()) return;
        int maxDim = std::max(dim[0], std::max(dim[1], dim[2]));
        float k = Clamp(64.0f / (float)maxDim, 0.15f, 2.0f) * opacity;
        auto alpha = [&](float a) { return (unsigned char)Clamp(a * k * 255.0f, 0.0f, 255.0f); };
        const Color green = { 60, 220, 110, alpha(0.022f) };
        const Color red = { 255, 45, 45, alpha(0.30f) };
        const Color yellow = { 255, 205, 40, alpha(0.022f) };
        const Color clear = { green.r, green.g, green.b, 0 };
        for (int a = 0; a < 3; a++) {
            int b = (a + 1) % 3, c = (a + 2) % 3;
            int W = dim[b], H = dim[c];
            for (int n = 0; n < dim[a]; n++) {
                Color* px = (Color*)MemAlloc((unsigned)(W * H * sizeof(Color)));
                bool any = false;
                for (int v = 0; v < H; v++)
                    for (int u = 0; u < W; u++) {
                        int co[3];
                        co[a] = n; co[b] = u; co[c] = v;
                        uint8_t cell = cells[((size_t)co[2] * dim[1] + co[1]) * dim[0] + co[0]];
                        Color col = (cell == 2 || cell == 4) ? red : (onlyOverlap ? clear : (cell == 1 ? green : (cell == 3 ? yellow : clear)));
                        any |= col.a > 0;
                        px[v * W + u] = col;
                    }
                if (!any) {
                    MemFree(px);
                    stack[a].push_back(Texture2D{});
                    continue;
                }
                Image img = { px, W, H, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 };
                Texture2D tex = LoadTextureFromImage(img);
                SetTextureFilter(tex, TEXTURE_FILTER_BILINEAR);
                SetTextureWrap(tex, TEXTURE_WRAP_CLAMP);
                UnloadImage(img);
                stack[a].push_back(tex);
            }
        }
    }

    void Viewport::BuildOverlapTextures() {
        m_OverlapDirty = false;
        const OverlapResult& r = m_OverlapResult;
        if (!r.ok) { UnloadStack(m_OverlapSlices); return; }
        BuildStack(m_OverlapSlices, r.cells, r.dim, m_OverlapOnly, m_OverlapOpacity);
    }

    static void DrawStack(const Camera3D& cam, const std::vector<Texture2D> (&stack)[3], Vector3 origin, float voxel, const int dim[3]) {
        if (stack[0].empty()) return;
        Vector3 fwd = Vector3Normalize(Vector3Subtract(cam.target, cam.position));
        const float* f = &fwd.x;
        int a = 0;
        for (int i = 1; i < 3; i++) if (fabsf(f[i]) > fabsf(f[a])) a = i;
        int b = (a + 1) % 3, c = (a + 2) % 3;
        const float o[3] = { origin.x, origin.y, origin.z };
        float half = voxel * 0.5f;
        float b0 = o[b] - half, b1 = o[b] + (dim[b] - 0.5f) * voxel;
        float c0 = o[c] - half, c1 = o[c] + (dim[c] - 0.5f) * voxel;
        const float pb[4] = { b0, b1, b1, b0 }, pc[4] = { c0, c0, c1, c1 };
        const float uv[4][2] = { { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 } };

        rlDrawRenderBatchActive();
        rlDisableDepthMask();
        rlDisableBackfaceCulling();
        rlSetBlendFactorsSeparate(RL_SRC_ALPHA, RL_ONE_MINUS_SRC_ALPHA, RL_ZERO, RL_ONE, RL_FUNC_ADD, RL_FUNC_ADD);
        BeginBlendMode(BLEND_CUSTOM_SEPARATE);
        int n = dim[a];
        for (int t = 0; t < n; t++) {
            int idx = f[a] > 0 ? n - 1 - t : t;
            const Texture2D& tex = stack[a][idx];
            if (tex.id == 0) continue;
            float pa = o[a] + idx * voxel;
            rlSetTexture(tex.id);
            rlBegin(RL_QUADS);
            rlColor4ub(255, 255, 255, 255);
            for (int k = 0; k < 4; k++) {
                float pt[3];
                pt[a] = pa; pt[b] = pb[k]; pt[c] = pc[k];
                rlTexCoord2f(uv[k][0], uv[k][1]);
                rlVertex3f(pt[0], pt[1], pt[2]);
            }
            rlEnd();
        }
        rlSetTexture(0);
        EndBlendMode();
        rlEnableBackfaceCulling();
        rlEnableDepthMask();
    }

    void Viewport::DrawOverlapVolume() const {
        const OverlapResult& r = m_OverlapResult;
        if (r.ok) DrawStack(m_Camera, m_OverlapSlices, r.origin, r.voxel, r.dim);
    }

    bool Viewport::FollowingOverlap() const {
        return m_OverlapSource && m_OverlapSource->m_ShowOverlap && m_HasModel &&
               m_OverlapSource->m_OverlapResult.ok && !m_OverlapSource->m_OverlapResult.sourceCells.empty();
    }

    void Viewport::UpdateFollowOverlap() {
        const Viewport* src = m_OverlapSource;
        if (!FollowingOverlap()) {
            if (!m_FollowSlices[0].empty()) UnloadStack(m_FollowSlices);
            m_FollowGeneration = 0;
            return;
        }
        if (m_FollowGeneration == src->m_OverlapGeneration && m_FollowOnly == src->m_OverlapOnly &&
            m_FollowOpacity == src->m_OverlapOpacity && !m_FollowSlices[0].empty())
            return;
        const OverlapResult& r = src->m_OverlapResult;
        BuildStack(m_FollowSlices, r.sourceCells, r.sourceDim, src->m_OverlapOnly, src->m_OverlapOpacity);
        m_FollowGeneration = src->m_OverlapGeneration;
        m_FollowOnly = src->m_OverlapOnly;
        m_FollowOpacity = src->m_OverlapOpacity;
    }

    void Viewport::DrawFollowOverlap() const {
        if (!FollowingOverlap()) return;
        const OverlapResult& r = m_OverlapSource->m_OverlapResult;
        DrawStack(m_Camera, m_FollowSlices, r.sourceOrigin, r.sourceVoxel, r.sourceDim);
    }

    void Viewport::DrawOverlapControls() {
        ImGui::SeparatorText("Fold overlap (right panel)");
        bool ready = m_Mirror && m_HasModel && m_Mirror->m_HasModel;
        if (!ready) ImGui::TextDisabled("Load a model into both panels to analyse the fold.");
        ImGui::BeginDisabled(!m_HasModel);
        ImGui::SliderFloat("Shape scale", &m_UserScale, 0.2f, 3.0f, "%.2fx");
        ImGui::SetItemTooltip("Resize the right shape around its centre. The skeleton and the folded\n"
                              "left model keep their size. Ctrl+click to type a value.");
        ImGui::SameLine();
        if (ImGui::SmallButton("1x")) m_UserScale = 1.0f;
        ImGui::EndDisabled();
        ImGui::BeginDisabled(!ready);
        ImGui::Checkbox("Show overlap volume", &m_ShowOverlap);
        ImGui::SetItemTooltip("Fold the left model along the posed skeleton and colour the inside of the\n"
                              "right model: green where the folded model fills it, yellow where it is empty,\n"
                              "red where parts overlap or poke out of the right shape.");
        ImGui::BeginDisabled(!m_ShowOverlap);
        ImGui::SliderInt("Resolution", &m_OverlapResolution, 24, 160, "%d voxels");
        ImGui::SetItemTooltip("Level set voxels along the longest side. Higher is sharper but slower.");
        ImGui::SliderFloat("Erosion", &m_OverlapErosion, 0.0f, 2.0f, "%.2f voxels");
        ImGui::SetItemTooltip("Shrinks the folded body this much before testing, so parts that only touch\n"
                              "at their skin don't count as overlapping.");
        if (ImGui::SliderFloat("Opacity", &m_OverlapOpacity, 0.1f, 2.0f, "%.2f")) m_OverlapDirty = true;
        if (ImGui::Checkbox("Overlap only", &m_OverlapOnly)) m_OverlapDirty = true;
        ImGui::SetItemTooltip("Hide green and yellow and show only the red.");
        ImGui::EndDisabled();
        ImGui::EndDisabled();

        ImGui::TextColored(ImVec4(0.25f, 0.88f, 0.43f, 1), "Green");
        ImGui::SameLine();
        ImGui::TextUnformatted("folded model");
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.16f, 1), "Yellow");
        ImGui::SameLine();
        ImGui::TextUnformatted("empty");
        ImGui::TextColored(ImVec4(1.0f, 0.18f, 0.18f, 1), "Red");
        ImGui::SameLine();
        ImGui::TextUnformatted("overlap, or outside the right shape");
        if (!m_ShowOverlap || !ready) return;
        if (m_Overlap.Busy()) ImGui::TextDisabled("Analysing...");
        if (!m_OverlapResult.message.empty()) {
            ImGui::TextWrapped("%s", m_OverlapResult.message.c_str());
            if (m_OverlapResult.ok) ImGui::TextDisabled("%.0f ms, %d x %d x %d voxels", m_OverlapResult.milliseconds,
                                                         m_OverlapResult.dim[0], m_OverlapResult.dim[1], m_OverlapResult.dim[2]);
        }
        if (!m_XrayModel) ImGui::TextDisabled("Turn on X-ray in the right panel to see the volume.");
    }
}
