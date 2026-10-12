#pragma once
#include "util/structs.hpp"
#include <vector>

namespace Bakugan {
    class Skeleton {
    public:
        Skeleton() {}
        ~Skeleton() {}
        int AddJoint(int parent, Vector3 worldPos, const std::string& name = "");
        void RemoveJoint(int index);
        void InsertJoint(int index, const Joint& j);
        int SplitBone(int child, int segments);
        std::vector<Matrix> WorldTransforms() const;
        Matrix ParentWorld(int index, const std::vector<Matrix>& world) const;
        Vector3 WorldToLocalOffset(int parent, Vector3 worldPos) const;
        void ResetPose();
    public:
        std::string GetName() const { return m_Name; }
        void SetName(std::string name) { m_Name = name; }
        Color GetColor() const { return m_Color; }
        void SetColor(Color c) { m_Color = c; }
        bool IsVisible() const { return m_Visible; }
        void SetVisibility(bool v) { m_Visible = v; }
        const std::vector<Joint>& Joints() const { return m_Joints; }
        std::vector<Joint>& JointsRef() { return m_Joints; }
    private:
        std::string m_Name = "Untitled";
        Vector3 m_Origin = { 0, 0, 0 };
        Color m_Color = ORANGE;
        int m_Next = 0;
        bool m_Visible = true;
        std::vector<Joint> m_Joints;
    };
}
