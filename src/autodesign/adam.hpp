#pragma once
// Adam (Kingma & Ba, 2015) over a set of parameters that can grow and shrink between steps.
//
// Every parameter comes with a key that stays the same for as long as the parameter exists (see params.hpp). Adam's
// memory (the running means of the gradient and of its square, and how many steps that parameter has had) is kept
// per key, so when the grammar adds parameters they start with fresh memory, parameters that are still there keep
// theirs, and the memory of parameters that are gone is dropped.
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Bakugan {
    class Adam {
    public:
        float lr = 1.0f;
        float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;

        // One step: x[i] moves against gradient g[i]; keys[i] names parameter i.
        void Step(const std::vector<uint64_t>& keys, std::vector<float>& x, const std::vector<float>& g) {
            for (size_t i = 0; i < x.size(); i++) {
                Slot& s = m_Slots[keys[i]];              // a new key gets a fresh slot
                s.t++;
                s.m = beta1 * s.m + (1.0f - beta1) * g[i];
                s.v = beta2 * s.v + (1.0f - beta2) * g[i] * g[i];
                float mHat = s.m / (1.0f - powf(beta1, (float)s.t));
                float vHat = s.v / (1.0f - powf(beta2, (float)s.t));
                x[i] -= lr * mHat / (sqrtf(vHat) + eps);
            }
            if (m_Slots.size() > keys.size()) Forget(keys);
        }

        size_t Slots() const { return m_Slots.size(); }

    private:
        // Drops the memory of parameters that are no longer there.
        void Forget(const std::vector<uint64_t>& keys) {
            std::unordered_set<uint64_t> live(keys.begin(), keys.end());
            for (auto it = m_Slots.begin(); it != m_Slots.end();)
                it = live.count(it->first) ? std::next(it) : m_Slots.erase(it);
        }

        struct Slot {
            float m = 0.0f, v = 0.0f;
            int t = 0;
        };
        std::unordered_map<uint64_t, Slot> m_Slots;
    };
}
