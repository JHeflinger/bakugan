#pragma once
#include "util/structs.hpp"
#include "ui/viewport.hpp"

namespace Bakugan {
    struct EditState {
        Skeleton skeleton;
        int leftSel = 0;
        std::unordered_map<int, Quaternion> pose;
        Vector3 offset = { 0, 0, 0 };
        int rightSel = -1;
    };

    class App {
    public:
        App();
        ~App();
        void Run();
    private:
        EditState Capture() const;
        void Restore(const EditState& st);
        static bool Interacting();
        void TrackHistory();
        void Undo();
        void Redo();
        void HandleUndoKeys();
        int AddToLibrary(const Asset& a);
        void LoadInto(Viewport& vp, int assetIndex);
        void HandleFileDrops();
        void AcceptAssetDrop(Viewport& vp);
        void DrawViewportWindow(Viewport& vp, ImVec2 pos, ImVec2 size, bool focusHint);
        void DrawLibrary();
        static bool ColorEdit(const char* id, Color* c);
        void DrawJointTree(int index);
        void DrawHierarchy();
        void DrawJointInspector();
        void DrawSidebar(float width, float height);
        void Frame();
    private:
        std::vector<Asset> m_Library;
        Viewport m_Left{ "Rig (edit skeleton)", true };
        Viewport m_Right{ "Morph (fit skeleton)", false };
        std::vector<EditState> m_UndoStack;
        std::vector<EditState> m_RedoStack;
        EditState m_Committed;
        Shader m_Shader = {};
        int m_ViewPositionLocation = -1;
        std::string m_Status = "Drag a model file onto either panel to begin.";
        int m_EulerJoint = -1;
        Quaternion m_EulerQuaternion = {};
        Vector3 m_EulerDegree = {};
    };
}
