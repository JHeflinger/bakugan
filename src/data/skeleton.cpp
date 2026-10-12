#include "skeleton.hpp"
#include "util/utils.hpp"
#include <raymath.h>

namespace Bakugan {
    int Skeleton::AddJoint(int parent, Vector3 worldPos, const std::string& jointName) {
        Joint j;
        j.id = m_Next++;
        j.parent = parent;
        j.offset = WorldToLocalOffset(parent, worldPos);
        j.name = jointName.empty() ? "joint_" + std::to_string(m_Joints.size()) : jointName;
        m_Joints.push_back(j);
        return (int)m_Joints.size() - 1;
    }

    void Skeleton::RemoveJoint(int index) {
        if (index < 0 || index >= (int)m_Joints.size()) return;
        std::vector<bool> doomed(m_Joints.size(), false);
        doomed[index] = true;
        for (size_t i = index + 1; i < m_Joints.size(); i++)
            if (m_Joints[i].parent >= 0 && doomed[m_Joints[i].parent]) doomed[i] = true;
        std::vector<int> remap(m_Joints.size(), -1);
        std::vector<Joint> kept;
        for (size_t i = 0; i < m_Joints.size(); i++) {
            if (doomed[i]) continue;
            remap[i] = (int)kept.size();
            kept.push_back(m_Joints[i]);
        }
        for (Joint& j : kept)
            if (j.parent >= 0) j.parent = remap[j.parent];
        m_Joints = std::move(kept);
    }

    void Skeleton::InsertJoint(int index, const Joint& j) {
        for (Joint& k : m_Joints)
            if (k.parent >= index) k.parent++;
        m_Joints.insert(m_Joints.begin() + index, j);
    }

    int Skeleton::SplitBone(int child, int segments) {
        if (child <= 0 || child >= (int)m_Joints.size() || segments < 2) return child;
        int parent = m_Joints[child].parent;
        std::vector<Matrix> world = WorldTransforms();
        Vector3 a = Utils::MatrixPosition(world[parent]), b = Utils::MatrixPosition(world[child]);
        std::string base = m_Joints[child].name;
        int first = child;
        for (int s = 1; s < segments; s++) {
            Joint j;
            j.id = m_Next++;
            j.parent = parent;
            j.offset = WorldToLocalOffset(parent, Vector3Lerp(a, b, (float)s / segments));
            char suffix[16];
            snprintf(suffix, sizeof(suffix), ".%03d", s);
            j.name = base + suffix;
            InsertJoint(child, j);
            parent = child++;
            m_Joints[child].parent = parent;
            m_Joints[child].offset = WorldToLocalOffset(parent, b);
        }
        return first;
    }

    std::vector<Matrix> Skeleton::WorldTransforms() const {
        std::vector<Matrix> world(m_Joints.size());
        Matrix root = MatrixTranslate(m_Origin.x, m_Origin.y, m_Origin.z);
        for (size_t i = 0; i < m_Joints.size(); i++) {
            const Joint& j = m_Joints[i];
            Matrix local = MatrixMultiply(QuaternionToMatrix(j.rotation),
                                          MatrixTranslate(j.offset.x, j.offset.y, j.offset.z));
            world[i] = MatrixMultiply(local, j.parent >= 0 ? world[j.parent] : root);
        }
        return world;
    }

    Matrix Skeleton::ParentWorld(int index, const std::vector<Matrix>& world) const {
        int p = m_Joints[index].parent;
        return p >= 0 ? world[p] : MatrixTranslate(m_Origin.x, m_Origin.y, m_Origin.z);
    }

    Vector3 Skeleton::WorldToLocalOffset(int parent, Vector3 worldPos) const {
        if (parent < 0) return Vector3Subtract(worldPos, m_Origin);
        return Vector3Transform(worldPos, MatrixInvert(WorldTransforms()[parent]));
    }

    void Skeleton::ResetPose() {
        for (Joint& j : m_Joints) j.rotation = QuaternionIdentity();
    }
}
