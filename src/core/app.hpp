#pragma once
#include "util/structs.hpp"
#include "ui/viewport.hpp"
#include "data/decompose.hpp"
#include "data/fold.hpp"
#include "data/restore.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace Bakugan {
    struct EditState {
        Design design;
        int selected = -1;
    };

    struct PartModel {
        SuperFrustum shape;          // the shape the mesh was built for (pose ignored)
        Model model = {};
        bool loaded = false;
    };

    struct RestoredModel {
        int partId = 0;
        Model model = {};
        TriMesh mesh;
    };

    enum class Pipeline {
        Idle,
        Decompose,
        Fold,
        Restore
    };

    class App {
    public:
        App();
        ~App();
        void Run();
    private:
        // app.cpp: frame, history, library, jobs
        void Frame();
        EditState Capture() const;
        void Restore(const EditState& st);
        static bool Interacting();
        void TrackHistory();
        void Undo();
        void Redo();
        void HandleKeys();
        int AddToLibrary(const Asset& a);
        void LoadAsset(int index);
        void HandleFileDrops();
        void AcceptAssetDrop(const Viewport& vp);
        void SetInputModel(Model m, const std::string& name);
        bool Busy() const;
        const Design& Shown() const;
        void PollJobs();
        void StartDecompose(bool refineOnly, int onlyIndex);
        void StartFold(bool keepTree);
        void StartRestore();
        void RunAll();
        void BuildTree();
        void AimSelected();
        void SnapPivotToContact(int part);
        void UpdateAnalysis();
        void ClearRestored();
        bool RestoredCurrent() const;
        void ExportParts();
        void SaveDesign();
        void LoadDesign();
        void Select(int part);
        int AddPart(const SuperFrustum& shape);
        void DeleteSelected();
        void DuplicateSelected();

        // panels.cpp: the inspector
        void DrawSidebar(float width, float height);
        void DrawLibrary();
        void DrawEditBar();
        void DrawAutomatic();
        void DrawManual();
        void DrawPartList();
        void DrawPartInspector();
        void DrawHingeInspector();
        void DrawSphereInspector();
        void DrawMetrics();
        void DrawFoldSettings(bool manual);
        void DrawRestorePanel();
        void DrawFilesPanel();
        bool DrawJob(const JobControl* control, bool busy, const char* id);

        // scene.cpp: the two viewports
        void DrawViewportWindow(Viewport& vp, ImVec2 pos, ImVec2 size, bool figure);
        void FigureToolbar();
        void BallToolbar();
        bool FigureOverlay(ImVec2 pos, float w, float h);
        bool BallOverlay(ImVec2 pos, float w, float h);
        void RenderFigure();
        void RenderBall();
        void DrawInput();
        void DrawSuperfrustums(const Design& d, const std::vector<Rigid>& xf, float alpha, int highlight);
        void DrawRestored(const Design& d, const std::vector<Rigid>& xf, float alpha);
        void DrawHinges(const Design& d, const std::vector<Rigid>& xf, float t);
        Model& PartMesh(const Part& p);
        void PurgePartMeshes(const Design& d);
        int Pick(Viewport& vp, const Design& d, const std::vector<Rigid>& xf);
        void UpdateFoldAnimation();
    private:
        Shader m_Shader = {};
        int m_ViewPositionLocation = -1;
        std::vector<Asset> m_Library;
        Model m_Input = {};
        bool m_HasInput = false;
        std::string m_InputName;
        Matrix m_InputTransform = {};
        BoundingBox m_InputBounds = { { -1, 0, -1 }, { 1, 2, 1 } };
        std::shared_ptr<const TriMesh> m_Mesh;
        std::shared_ptr<const ShapeAnalysis> m_Shape;

        Design m_Design;
        int m_Selected = -1;
        Mode m_Mode = Mode::Automatic;
        EditTarget m_Target = EditTarget::Part;
        GizmoOp m_GizmoOp = GizmoOp::Translate;
        bool m_LocalGizmo = true;

        Viewport m_Figure{ "Figure  (open pose)" };
        Viewport m_Ball{ "Ball  (fold preview)" };

        std::vector<EditState> m_UndoStack;
        std::vector<EditState> m_RedoStack;
        EditState m_Committed;

        Job<DecomposeResult> m_DecomposeJob;
        Job<FoldResult> m_FoldJob;
        Job<RestoreResult> m_RestoreJob;
        DecomposeParams m_DecomposeParams;
        FoldParams m_FoldParams;
        RestoreParams m_RestoreParams;
        Pipeline m_Pipeline = Pipeline::Idle;
        bool m_RunAll = false;
        std::vector<float> m_LastHistory;
        std::shared_ptr<const Design> m_Preview;
        FoldParams m_ModelParams;

        std::vector<RestoredModel> m_Restored;
        Design m_RestoredFor;
        std::unordered_map<int, PartModel> m_PartModels;

        FoldModel m_FoldModel;
        FoldState m_Analysis;
        FoldMetrics m_Metrics;
        FoldVisual m_Visual;
        bool m_AnalysisValid = false;
        bool m_ShapesDirty = true;

        bool m_ShowInput = true;
        float m_InputAlpha = 0.22f;
        bool m_ShowParts = true;
        float m_PartsAlpha = 0.55f;
        bool m_ShowHinges = true;
        bool m_ShowPiecesInFigure = false;
        bool m_ShowPieces = true;
        bool m_ShowSphere = true;
        bool m_ShowShell = false;
        bool m_ShowLost = false;
        bool m_ShowGrid = true;

        float m_Fold = 1.0f;
        float m_FoldVelocity = 0.0f;
        int m_FoldAnim = 0;             // 0 none, 1 spring open, 2 close
        bool m_Loop = false;
        float m_LoopTimer = 0.0f;

        int m_EulerPart = -1;
        Quaternion m_EulerQuaternion = {};
        Vector3 m_EulerDegrees = {};

        std::string m_Status = "Pick a model from the library or drop a file onto a panel to begin.";
        char m_DesignPath[256] = "bakugan_design.txt";
        char m_ExportPath[256] = "bakugan_parts";
    };
}
