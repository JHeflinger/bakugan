#pragma once
#include "util/structs.hpp"
#include "ui/viewport.hpp"
#include "ui/raymarch.hpp"
#include "data/build.hpp"
#include "data/collide.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Bakugan {
    // What the inspector and the gizmo work on: some shapes, one group, or one hinge.
    struct Selection {
        std::vector<int> shapes;     // shape ids
        int group = -1;              // group id
        int hinge = -1;              // hinge id
        bool Empty() const { return shapes.empty() && group < 0 && hinge < 0; }
        bool operator==(const Selection& o) const { return shapes == o.shapes && group == o.group && hinge == o.hinge; }
    };

    struct EditState {
        Design design;
        Selection selection;
    };

    struct ShapeModel {
        SuperFrustum shape;          // the shape the mesh was built for (pose ignored)
        TriMesh mesh;                // local frame
        Model model = {};
        bool loaded = false;
    };

    struct PartModel {
        int body = -1;               // group id, -1 for a separate pin
        int pinHinge = -1;
        std::string key;             // BuiltPart::key: an unchanged part keeps its GPU model
        Model model = {};
        size_t triangles = 0;
    };

    enum class Preset {
        Box,
        Cylinder,
        Cone,
        Ball,
        Wedge,
        Tube,
        Arc
    };

    class App {
    public:
        App();
        ~App();
        void Run();
    private:
        // app.cpp: frame, history, files, jobs, editing
        void Frame();
        EditState Capture() const;
        void Restore(const EditState& st);
        static bool Interacting();
        void TrackHistory();
        void Undo();
        void Redo();
        void HandleKeys();
        void SetDesign(const Design& d, const std::string& status);
        void LoadTemplate(int index);
        bool OpenFile(const std::string& path);
        void HandleFileDrops();
        bool FileDialog(bool save, const char* filters, const std::string& start, std::string* out);
        void BrowseDesign(bool save);
        void BrowseExport();
        void BrowseReference();
        void LoadReference(const std::string& path);
        void SaveDesign();
        void LoadDesign();
        void PollJobs();
        void UpdateJobs();
        void StartExport();
        bool PartsCurrent() const;
        bool SweepCurrent() const;
        const SweepStep* CurrentStep() const;
        void UpdateAnimation();
        void Play(int direction);                    // +1 close, -1 open
        void ShowMoment(float t, Motion m);          // stop and jump the output to a moment
        static const char* MotionWord(Motion m) { return m == Motion::Opening ? "opening" : "closing"; }

        void ClearSelection();
        void SelectShape(int id, bool additive);
        void SelectGroup(int id);
        void SelectHinge(int id);
        void SanitizeSelection();
        int SelectedShape() const;           // index when exactly one shape is selected
        int SelectedGroup() const;           // index
        int SelectedHinge() const;           // index
        std::vector<int> SelectedShapeIndices() const;   // the selected shapes, or every shape of the selected group
        int ContextGroup() const;            // group id new shapes go into (-1: a new body)
        void AddPreset(Preset p);
        void NewBody();
        void GroupSelection();
        void UngroupSelection();
        void DeleteSelection();
        void DuplicateSelection();
        void ToggleCut();
        bool GroupHidden(int groupId) const;          // it or an enclosing group is hidden
        bool ShapeHidden(int shapeIndex) const;       // it or a group it is in is hidden
        void ToggleHidden(int id);
        void HideSelection();
        int AddHingeBetween(int parent, int child);
        void PlaceHinge(int hinge);
        void FrameView(Viewport& vp);

        // panels.cpp: the sidebar
        void DrawSidebar(float width, float height);
        void DrawDesignHeader();
        void DrawEditBar();
        void DrawModelTab();
        void DrawAddButtons();
        void DrawTree();
        void DrawShapeInspector();
        void DrawGroupInspector();
        void DrawReferencePanel();
        void DrawHingesTab();
        void DrawHingeList();
        void DrawHingeInspector();
        void DrawKnuckleToggles(int hinge);
        void DrawCollisionPanel();
        void DrawPrintTab();
        void DrawFilesPanel();
        bool DrawJob(const JobControl* control, bool busy, const char* id, bool stoppable);
        std::string ItemName(int shapeId, int hingeId) const;

        // scene.cpp: the two viewports
        void DrawViewportWindow(Viewport& vp, ImVec2 pos, ImVec2 size, bool design);
        void DesignToolbar();
        void OutputToolbar();
        void DrawTimeline(float width);
        bool DesignOverlay(ImVec2 pos, float w, float h);
        bool OutputOverlay(ImVec2 pos, float w, float h);
        bool GroupGizmo(const float* view, const float* proj);
        void RenderDesign();
        void RenderOutput();
        void DrawReference(bool solidPass);
        void DrawShapeGhosts();
        Matrix ReferenceMatrix() const;
        BoundingBox ReferenceBounds() const;
        void DrawShapes(const std::vector<Rigid>& xf, bool output, float alpha);
        void DrawShapesExact(const std::vector<Rigid>& xf, bool output, float alpha);
        bool Exact() const { return m_Exact && m_Sdf.Ok(); }
        SdfView ViewOf(const Viewport& vp) const;
        void SyncExact();
        void DrawHingeHardware(const std::vector<Rigid>& xf, bool output);
        void DrawCollisions();
        void DrawCollisionMarks();
        ShapeModel& ShapeMesh(const Shape& s);
        void PurgeShapeMeshes();
        void ClearParts();
        int PickShape(Viewport& vp, const std::vector<Rigid>& xf, bool cuts);
    private:
        Shader m_Shader = {};
        SdfRenderer m_Sdf;
        bool m_Exact = true;                          // raymarch the fields instead of drawing meshes
        bool m_ExactSynced = false;
        Design m_ExactFor;
        int m_ViewPositionLocation = -1;

        Design m_Design;
        Selection m_Sel;
        GizmoOp m_GizmoOp = GizmoOp::Translate;
        PickMode m_PickMode = PickMode::Shape;
        bool m_LocalGizmo = false;
        bool m_Snap = false;
        float m_SnapMove = 1.0f;
        float m_SnapAngle = 15.0f;

        Viewport m_DesignView{ "Design  (as modelled)" };
        Viewport m_OutputView{ "Output  (parts in motion)" };

        std::vector<EditState> m_UndoStack;
        std::vector<EditState> m_RedoStack;
        EditState m_Committed;

        // group / multi-shape gizmo: transform applied to a snapshot taken when the drag started
        bool m_GroupDrag = false;
        float m_GroupMatrix[16] = {};
        Matrix m_GroupStart = {};
        Design m_DragStart;
        std::vector<int> m_DragShapes;
        std::vector<int> m_DragHinges;

        // parts, collisions, export
        Job<BuildResult> m_BuildJob;
        Job<SweepResult> m_SweepJob;
        Job<ExportResult> m_ExportJob;
        std::shared_ptr<const std::vector<BuiltPart>> m_Parts;
        std::vector<BuiltPart> m_SpareParts;          // finished by a build that was stopped: reused by the next one
        std::shared_ptr<SweepCache> m_SweepCache = std::make_shared<SweepCache>();
        double m_BuildStart = 0.0, m_SweepStart = 0.0;
        std::string m_BuildStats, m_SweepStats;       // how long the last update took, and how much was reused
        Design m_BuildingFor;
        float m_BuildingVoxel = 0.0f;
        Design m_SweepingFor;
        SweepParams m_SweepingParams;
        std::vector<PartModel> m_PartModels;
        Design m_PartsFor;
        float m_PartsVoxel = 0.0f;
        bool m_HasParts = false;
        int m_PreviewResolution = 220;
        SweepResult m_Sweep;
        bool m_HasSweep = false;
        SweepParams m_SweepParams;
        ExportParams m_ExportParams;
        std::vector<std::string> m_ExportedFiles;

        std::unordered_map<int, ShapeModel> m_ShapeModels;
        std::unordered_set<int> m_Hidden;             // shape and group ids hidden in the views (not part of the design)

        // reference model
        Model m_Reference = {};
        bool m_HasReference = false;
        std::string m_ReferenceName;
        bool m_ShowReference = true;
        float m_ReferenceAlpha = 0.25f;
        float m_ReferenceScale = 1.0f;
        Vector3 m_ReferenceOffset = { 0, 0, 0 };
        Vector3 m_ReferenceRotation = { 0, 0, 0 };   // degrees, about the mesh's own centre
        BoundingBox m_ReferenceBox = {};              // as loaded (unscaled)
        ReferenceMode m_ReferenceMode = ReferenceMode::Solid;
        Color m_ReferenceColor = { 196, 198, 206, 255 };
        bool m_ReferenceInOutput = false;
        bool m_ReferenceOpen = false;                 // open the panel once after a load

        // display
        bool m_ShowCuts = true;
        float m_ShapesAlpha = 1.0f;
        bool m_ShowHinges = true;
        bool m_ShowGrid = true;
        bool m_ShowBuilt = true;
        bool m_ShowContacts = true;
        bool m_GhostOpen = false;

        // motion: t = 0 open, 1 closed
        float m_T = 0.0f;
        int m_Anim = 0;              // +1 closing, -1 opening
        Motion m_Motion = Motion::Closing;   // the schedule the output follows (picked when a play starts at an end)
        bool m_Loop = false;
        float m_LoopTimer = 0.0f;
        float m_Duration = 2.5f;     // seconds for the whole motion
        bool m_Ease = true;

        int m_EulerShape = -1;
        Quaternion m_EulerQuaternion = {};
        Vector3 m_EulerDegrees = {};
        int m_TemplateIndex = 0;
        int m_NewHingeParent = -1;
        int m_NewHingeChild = -1;
        int m_RenameId = -1;

        std::string m_Status = "Start from a template, or add shapes in the Model tab.";
        char m_DesignPath[256] = "hinge_design.txt";
        char m_ExportPath[256] = "bakugan_parts";
        char m_ReferencePath[512] = "";
    };
}
