#pragma once
#include <cmath>
#include <vector>

namespace Bakugan {
    // Adam with a learning rate per parameter.
    struct Adam {
        std::vector<float> m, v;
        int t = 0;

        void Resize(size_t n) { m.assign(n, 0.0f); v.assign(n, 0.0f); t = 0; }

        void Step(float* params, const float* grad, const float* lr, size_t n) {
            if (m.size() != n) Resize(n);
            t++;
            const float b1 = 0.9f, b2 = 0.999f;
            float c1 = 1.0f - powf(b1, (float)t), c2 = 1.0f - powf(b2, (float)t);
            for (size_t i = 0; i < n; i++) {
                m[i] = b1 * m[i] + (1 - b1) * grad[i];
                v[i] = b2 * v[i] + (1 - b2) * grad[i] * grad[i];
                float mh = m[i] / c1, vh = v[i] / c2;
                params[i] -= lr[i] * mh / (sqrtf(vh) + 1e-12f);
            }
        }
    };

    inline float Sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
}
