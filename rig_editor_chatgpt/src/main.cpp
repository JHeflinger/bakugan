#include "raylib.h"
#include "raymath.h"
#include "rlImGui.h"
#include "imgui.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr int kInitialWidth = 1440;
constexpr int kInitialHeight = 900;
constexpr float kMinCameraDistance = 1.0f;
constexpr float kMaxCameraDistance = 80.0f;

struct ModelInstance {
    Model model{};
    std::string path;
    Vector3 position{0.0f, 0.0f, 0.0f};
    float scale = 1.0f;
};

struct Bone {
    int id = -1;
    int parent = -1;
    std::string name;
    Vector3 localPosition{0.0f, 0.0f, 0.0f};
};

struct Skeleton {
    std::vector<Bone> bones;
    int nextId = 0;
    int selected = -1;

    int AddBone(const std::string& name, int parent, Vector3 localPosition) {
        Bone bone;
        bone.id = nextId++;
        bone.parent = parent;
        bone.name = name;
        bone.localPosition = localPosition;
        bones.push_back(bone);
        selected = bone.id;
        return bone.id;
    }

    Bone* Find(int id) {
        for (Bone& bone : bones) {
            if (bone.id == id) return &bone;
        }
        return nullptr;
    }

    const Bone* Find(int id) const {
        for (const Bone& bone : bones) {
            if (bone.id == id) return &bone;
        }
        return nullptr;
    }

    Vector3 WorldPosition(int id) const {
        const Bone* bone = Find(id);
        if (!bone || bone->parent < 0) return bone ? bone->localPosition : Vector3{0};
        return Vector3Add(WorldPosition(bone->parent), bone->localPosition);
    }

    void DeleteSelected() {
        if (selected < 0) return;

        const int deleted = selected;
        const Bone* deletedBone = Find(deleted);
        if (!deletedBone) return;

        const int newParent = deletedBone->parent;
        // Reparent direct children while preserving their world-space positions.
        // This avoids unexpected jumps when the hierarchy is edited.
        for (Bone& bone : bones) {
            if (bone.parent == deleted) {
                const Vector3 childWorld = WorldPosition(bone.id);
                bone.parent = newParent;
                const Vector3 newParentWorld = newParent >= 0 ? WorldPosition(newParent)
                                                               : Vector3{0.0f, 0.0f, 0.0f};
                bone.localPosition = Vector3Subtract(childWorld, newParentWorld);
            }
        }

        bones.erase(
            std::remove_if(bones.begin(), bones.end(), [deleted](const Bone& bone) {
                return bone.id == deleted;
            }),
            bones.end());

        selected = bones.empty() ? -1 : bones.back().id;
    }
};

struct ViewportState {
    std::string title;
    Camera3D camera{};
    float yaw = 0.65f;
    float pitch = 0.28f;
    float distance = 10.0f;
    RenderTexture2D target{};
    int targetWidth = 0;
    int targetHeight = 0;
    ImVec2 imagePos{0, 0};
    ImVec2 imageSize{0, 0};
    std::vector<ModelInstance> models;
};

struct PendingSkeletonAction {
    bool valid = false;
    Vector3 world{0.0f, 0.0f, 0.0f};
};

bool IsModelExtension(const char* path) {
    const char* dot = GetFileExtension(path);
    if (!dot) return false;

    std::string ext(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    return ext == ".obj" || ext == ".gltf" || ext == ".glb" ||
           ext == ".iqm" || ext == ".m3d";
}

void ConfigureCamera(ViewportState& view) {
    const float cp = std::cos(view.pitch);
    view.camera.position = {
        view.camera.target.x + std::sin(view.yaw) * cp * view.distance,
        view.camera.target.y + std::sin(view.pitch) * view.distance,
        view.camera.target.z + std::cos(view.yaw) * cp * view.distance
    };
    view.camera.up = {0.0f, 1.0f, 0.0f};
    view.camera.fovy = 45.0f;
    view.camera.projection = CAMERA_PERSPECTIVE;
}

void InitViewport(ViewportState& view, const char* title, Vector3 target) {
    view.title = title;
    view.camera.target = target;
    ConfigureCamera(view);
}

void EnsureRenderTarget(ViewportState& view, int width, int height) {
    width = std::max(width, 64);
    height = std::max(height, 64);

    if (view.target.id != 0 && view.targetWidth == width && view.targetHeight == height) {
        return;
    }

    if (view.target.id != 0) {
        UnloadRenderTexture(view.target);
        view.target = {};
    }

    view.target = LoadRenderTexture(width, height);
    view.targetWidth = width;
    view.targetHeight = height;
}

void DrawGridAndAxes(float extent = 20.0f) {
    DrawGrid(static_cast<int>(extent * 2.0f), 1.0f);
    DrawLine3D({-extent, 0.0f, 0.0f}, {extent, 0.0f, 0.0f}, RED);
    DrawLine3D({0.0f, -extent, 0.0f}, {0.0f, extent, 0.0f}, GREEN);
    DrawLine3D({0.0f, 0.0f, -extent}, {0.0f, 0.0f, extent}, BLUE);
}

void DrawModelInstances(const std::vector<ModelInstance>& models) {
    for (const ModelInstance& instance : models) {
        DrawModelEx(
            instance.model,
            instance.position,
            {0.0f, 1.0f, 0.0f},
            0.0f,
            {instance.scale, instance.scale, instance.scale},
            WHITE);
    }
}

void DrawSkeleton(const Skeleton& skeleton) {
    for (const Bone& bone : skeleton.bones) {
        const Vector3 world = skeleton.WorldPosition(bone.id);
        const bool selected = bone.id == skeleton.selected;

        if (bone.parent >= 0) {
            const Vector3 parentWorld = skeleton.WorldPosition(bone.parent);
            DrawCylinderEx(
                parentWorld,
                world,
                selected ? 0.075f : 0.055f,
                selected ? 0.075f : 0.055f,
                8,
                selected ? YELLOW : ORANGE);
        }

        DrawSphere(world, selected ? 0.16f : 0.12f, selected ? YELLOW : SKYBLUE);
    }
}

Ray MakeViewportRay(const ViewportState& view, Vector2 localPosition) {
    return GetScreenToWorldRayEx(localPosition, view.camera, view.targetWidth, view.targetHeight);
}

bool IntersectZPlane(Ray ray, Vector3* result) {
    constexpr float epsilon = 0.00001f;
    if (std::fabs(ray.direction.z) < epsilon) return false;

    const float t = -ray.position.z / ray.direction.z;
    if (t < 0.0f) return false;

    *result = Vector3Add(ray.position, Vector3Scale(ray.direction, t));
    return true;
}

bool GetViewportLocalMouse(const ViewportState& view, Vector2* localPosition) {
    const Vector2 mouse = GetMousePosition();
    const float x = mouse.x - view.imagePos.x;
    const float y = mouse.y - view.imagePos.y;

    if (x < 0.0f || y < 0.0f || x >= view.imageSize.x || y >= view.imageSize.y) {
        return false;
    }

    if (view.imageSize.x <= 1.0f || view.imageSize.y <= 1.0f) return false;

    *localPosition = {
        x * static_cast<float>(view.targetWidth) / view.imageSize.x,
        y * static_cast<float>(view.targetHeight) / view.imageSize.y
    };
    return true;
}

bool RayHitCreationPlane(const ViewportState& view, Vector3* result) {
    Vector2 local;
    if (!GetViewportLocalMouse(view, &local)) return false;
    return IntersectZPlane(MakeViewportRay(view, local), result);
}

void SelectBoneFromMouse(const ViewportState& view, Skeleton& skeleton) {
    Vector2 local;
    if (!GetViewportLocalMouse(view, &local)) return;

    int bestId = -1;
    float bestDistance = 10.0f;

    for (const Bone& bone : skeleton.bones) {
        const Vector3 world = skeleton.WorldPosition(bone.id);
        const Vector2 screen = GetWorldToScreenEx(world, view.camera, view.targetWidth, view.targetHeight);
        const float dx = screen.x - local.x;
        const float dy = screen.y - local.y;
        const float d2 = dx * dx + dy * dy;
        if (d2 < bestDistance * bestDistance) {
            bestDistance = std::sqrt(d2);
            bestId = bone.id;
        }
    }

    skeleton.selected = bestId;
}

void UpdateCameraInteraction(ViewportState& view, bool hovered) {
    if (!hovered || ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId)) return;

    const ImGuiIO& io = ImGui::GetIO();
    if (io.WantTextInput) return;

    const float wheel = io.MouseWheel;
    if (std::fabs(wheel) > 0.0f) {
        view.distance *= std::pow(0.85f, wheel);
        view.distance = std::clamp(view.distance, kMinCameraDistance, kMaxCameraDistance);
    }

    if (ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
        const ImVec2 delta = io.MouseDelta;
        view.yaw -= delta.x * 0.01f;
        view.pitch -= delta.y * 0.01f;
        view.pitch = std::clamp(view.pitch, -1.45f, 1.45f);
    }

    ConfigureCamera(view);
}

void ProcessDroppedModels(ViewportState& rig, ViewportState& reference) {
    if (!IsFileDropped()) return;

    const Vector2 mouse = GetMousePosition();
    auto inside = [mouse](const ViewportState& view) {
        return mouse.x >= view.imagePos.x &&
               mouse.y >= view.imagePos.y &&
               mouse.x < view.imagePos.x + view.imageSize.x &&
               mouse.y < view.imagePos.y + view.imageSize.y;
    };

    ViewportState* destination = nullptr;
    if (inside(rig)) destination = &rig;
    else if (inside(reference)) destination = &reference;

    FilePathList dropped = LoadDroppedFiles();
    if (destination) {
        for (unsigned int i = 0; i < dropped.count; ++i) {
            const char* path = dropped.paths[i];
            if (!IsModelExtension(path)) continue;

            Model model = LoadModel(path);
            if (model.meshCount <= 0) {
                UnloadModel(model);
                continue;
            }

            ModelInstance instance;
            instance.model = model;
            instance.path = path;
            instance.position = {0.0f, 0.0f, 0.0f};
            instance.scale = 1.0f;
            destination->models.push_back(std::move(instance));
        }
    }

    UnloadDroppedFiles(dropped);
}

void DrawViewportWindow(ViewportState& view,
                        Skeleton* skeleton,
                        PendingSkeletonAction* pendingAction,
                        bool allowRigging) {
    ImGui::Begin(view.title.c_str());

    const ImVec2 available = ImGui::GetContentRegionAvail();
    const int width = std::max(64, static_cast<int>(available.x));
    const int height = std::max(64, static_cast<int>(available.y));
    EnsureRenderTarget(view, width, height);

    const ImVec2 imagePos = ImGui::GetCursorScreenPos();
    const ImVec2 imageSize = available;
    view.imagePos = imagePos;
    view.imageSize = imageSize;

    // The image is deliberately rendered at the same pixel dimensions as the ImGui item.
    rlImGuiImageRect(
        &view.target.texture,
        width,
        height,
        {0.0f, 0.0f, static_cast<float>(view.target.texture.width),
         -static_cast<float>(view.target.texture.height)});

    const bool hovered = ImGui::IsItemHovered();

    if (hovered) {
        UpdateCameraInteraction(view, true);

        if (allowRigging && skeleton) {
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                SelectBoneFromMouse(view, *skeleton);
            }

            if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
                Vector3 world;
                if (RayHitCreationPlane(view, &world)) {
                    pendingAction->valid = true;
                    pendingAction->world = world;
                }
                ImGui::OpenPopup("SkeletonContext");
            }
        }
    }

    ImGui::SetCursorScreenPos({imagePos.x, imagePos.y + 5.0f});
    ImGui::TextDisabled("%s  |  files: %zu", allowRigging ? "Rig viewport" : "Reference viewport", view.models.size());

    if (allowRigging && skeleton && ImGui::BeginPopup("SkeletonContext")) {
        const bool canAddChild = skeleton->selected >= 0;
        if (ImGui::MenuItem("Add Root Bone")) {
            if (pendingAction->valid) {
                char name[64];
                std::snprintf(name, sizeof(name), "Bone_%02zu", skeleton->bones.size() + 1);
                skeleton->AddBone(name, -1, pendingAction->world);
            }
            pendingAction->valid = false;
        }

        if (ImGui::MenuItem("Add Child Bone", nullptr, false, canAddChild)) {
            if (pendingAction->valid) {
                const int parent = skeleton->selected;
                const Vector3 parentWorld = skeleton->WorldPosition(parent);
                const Vector3 local = Vector3Subtract(pendingAction->world, parentWorld);
                char name[64];
                std::snprintf(name, sizeof(name), "Bone_%02zu", skeleton->bones.size() + 1);
                skeleton->AddBone(name, parent, local);
            }
            pendingAction->valid = false;
        }

        ImGui::Separator();
        if (ImGui::MenuItem("Delete Selected", "Delete", false, skeleton->selected >= 0)) {
            skeleton->DeleteSelected();
            pendingAction->valid = false;
        }

        ImGui::Separator();
        ImGui::TextDisabled("Click a joint to select it.");
        ImGui::TextDisabled("Middle-drag orbit  |  Wheel zoom");
        ImGui::EndPopup();
    }

    ImGui::End();
}

void DrawSkeletonPanel(Skeleton& skeleton) {
    ImGui::Begin("Skeleton");

    if (skeleton.bones.empty()) {
        ImGui::TextWrapped("Right-click the rig viewport to add the first bone.");
    } else {
        for (const Bone& bone : skeleton.bones) {
            const bool isSelected = bone.id == skeleton.selected;
            ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf |
                                       ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                       (isSelected ? ImGuiTreeNodeFlags_Selected : 0);
            ImGui::TreeNodeEx(reinterpret_cast<void*>(static_cast<intptr_t>(bone.id)), flags, "%s", bone.name.c_str());
            if (ImGui::IsItemClicked()) skeleton.selected = bone.id;
        }
    }

    ImGui::Separator();

    Bone* selected = skeleton.Find(skeleton.selected);
    if (selected) {
        ImGui::TextUnformatted("Bone properties");
        char nameBuffer[128];
        std::snprintf(nameBuffer, sizeof(nameBuffer), "%s", selected->name.c_str());
        if (ImGui::InputText("Name", nameBuffer, sizeof(nameBuffer))) {
            selected->name = nameBuffer;
        }

        ImGui::InputFloat3("Local Position", &selected->localPosition.x, "%.3f");
        ImGui::Text("Parent: %s", selected->parent < 0 ? "<root>" : skeleton.Find(selected->parent)->name.c_str());

        if (ImGui::Button("Delete Bone")) {
            skeleton.DeleteSelected();
        }
    }

    ImGui::End();
}

void DrawFileDropStatus(const ViewportState& rig, const ViewportState& reference) {
    ImGui::Begin("Instructions");
    ImGui::TextUnformatted("Rig Editor");
    ImGui::Separator();
    ImGui::TextWrapped("Drag OBJ, GLTF/GLB, IQM, or M3D files from your file manager into either viewport.");
    ImGui::TextWrapped("The viewport under the mouse receives the dropped models.");
    ImGui::Spacing();
    ImGui::Text("Rig models: %zu", rig.models.size());
    ImGui::Text("Reference models: %zu", reference.models.size());
    ImGui::Spacing();
    ImGui::TextDisabled("Left viewport: skeleton editing");
    ImGui::TextDisabled("Right viewport: model/reference view");
    ImGui::End();
}

void RenderScene(ViewportState& view, const Skeleton* skeleton) {
    BeginTextureMode(view.target);
    ClearBackground({22, 23, 26, 255});

    BeginMode3D(view.camera);
    DrawGridAndAxes();
    DrawModelInstances(view.models);
    if (skeleton) DrawSkeleton(*skeleton);
    EndMode3D();

    EndTextureMode();
}

} // namespace

int main() {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
    InitWindow(kInitialWidth, kInitialHeight, "Rig Editor - raylib + Dear ImGui");
    SetTargetFPS(144);

    rlImGuiSetup(true);

    ViewportState rigViewport;
    ViewportState referenceViewport;
    InitViewport(rigViewport, "Rig Viewport", {0.0f, 2.0f, 0.0f});
    InitViewport(referenceViewport, "Reference Viewport", {0.0f, 2.0f, 0.0f});

    Skeleton skeleton;
    PendingSkeletonAction pendingAction;

    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground({16, 17, 20, 255});

        // Render into the previous frame's target dimensions. The viewport windows update those
        // dimensions before the next frame, so resizing naturally converges within one frame.
        if (rigViewport.target.id != 0) RenderScene(rigViewport, &skeleton);
        if (referenceViewport.target.id != 0) RenderScene(referenceViewport, nullptr);

        rlImGuiBegin();

        ImGui::SetNextWindowPos({10, 10}, ImGuiCond_Once);
        ImGui::SetNextWindowSize({430, 185}, ImGuiCond_Once);
        DrawFileDropStatus(rigViewport, referenceViewport);

        ImGui::SetNextWindowPos({10, 205}, ImGuiCond_Once);
        ImGui::SetNextWindowSize({700, 665}, ImGuiCond_Once);
        DrawViewportWindow(rigViewport, &skeleton, &pendingAction, true);

        ImGui::SetNextWindowPos({720, 205}, ImGuiCond_Once);
        ImGui::SetNextWindowSize({700, 665}, ImGuiCond_Once);
        DrawViewportWindow(referenceViewport, nullptr, &pendingAction, false);

        ImGui::SetNextWindowPos({1050, 10}, ImGuiCond_Once);
        ImGui::SetNextWindowSize({370, 185}, ImGuiCond_Once);
        DrawSkeletonPanel(skeleton);

        rlImGuiEnd();
        EndDrawing();

        // Process drops after the ImGui items have established their current rectangles.
        // The mouse position at drop time determines the destination viewport.
        ProcessDroppedModels(rigViewport, referenceViewport);
    }

    for (ModelInstance& model : rigViewport.models) UnloadModel(model.model);
    for (ModelInstance& model : referenceViewport.models) UnloadModel(model.model);
    if (rigViewport.target.id != 0) UnloadRenderTexture(rigViewport.target);
    if (referenceViewport.target.id != 0) UnloadRenderTexture(referenceViewport.target);

    rlImGuiShutdown();
    CloseWindow();
    return 0;
}
