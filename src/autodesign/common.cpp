#include "common.hpp"
#include <algorithm>
#include <cfloat>
#include <cmath>

namespace Bakugan::Auto {
    Vector3 RandomUnit(Rng& r) {
        for (;;) {
            Vector3 v = { Normal(r), Normal(r), Normal(r) };
            float l = Vector3Length(v);
            if (l > 1e-4f) return Vector3Scale(v, 1.0f / l);
        }
    }

    Vector3 RandomInBall(Rng& r) {
        Vector3 p;
        do p = { Uniform(r, -1, 1), Uniform(r, -1, 1), Uniform(r, -1, 1) };
        while (Vector3LengthSqr(p) > 1.0f);
        return p;
    }

    Quaternion RandomRotation(Rng& r) {
        Quaternion q = { Normal(r), Normal(r), Normal(r), Normal(r) };
        return QuaternionNormalize(q);
    }

    float WrapAngle(float a) {
        while (a > PI) a -= 2.0f * PI;
        while (a <= -PI) a += 2.0f * PI;
        return a;
    }

    float BoxDistance(Vector3 p, const BoundingBox& b) {
        float dx = fmaxf(fmaxf(b.min.x - p.x, p.x - b.max.x), 0.0f);
        float dy = fmaxf(fmaxf(b.min.y - p.y, p.y - b.max.y), 0.0f);
        float dz = fmaxf(fmaxf(b.min.z - p.z, p.z - b.max.z), 0.0f);
        return sqrtf(dx * dx + dy * dy + dz * dz);
    }

    Vector3 InBox(Vector3 lo, Vector3 ext, Vector3 u) { return { lo.x + u.x * ext.x, lo.y + u.y * ext.y, lo.z + u.z * ext.z }; }

    Quaternion FromAxes(Vector3 x, Vector3 y, Vector3 z) {
        float m00 = x.x, m10 = x.y, m20 = x.z, m01 = y.x, m11 = y.y, m21 = y.z, m02 = z.x, m12 = z.y, m22 = z.z;
        float tr = m00 + m11 + m22;
        Quaternion q;
        if (tr > 0.0f) {
            float s = sqrtf(tr + 1.0f) * 2.0f;
            q = { (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25f * s };
        } else if (m00 > m11 && m00 > m22) {
            float s = sqrtf(1.0f + m00 - m11 - m22) * 2.0f;
            q = { 0.25f * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s };
        } else if (m11 > m22) {
            float s = sqrtf(1.0f + m11 - m00 - m22) * 2.0f;
            q = { (m01 + m10) / s, 0.25f * s, (m12 + m21) / s, (m02 - m20) / s };
        } else {
            float s = sqrtf(1.0f + m22 - m00 - m11) * 2.0f;
            q = { (m02 + m20) / s, (m12 + m21) / s, 0.25f * s, (m10 - m01) / s };
        }
        return QuaternionNormalize(q);
    }

    // Eigen decomposition of a symmetric 3x3 matrix (Jacobi rotations), largest eigenvalue first.
    static void Eigen3(const double in[3][3], double vals[3], Vector3 vecs[3]) {
        double A[3][3], V[3][3] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) A[i][j] = in[i][j];
        for (int sweep = 0; sweep < 32; sweep++) {
            int p = 0, q = 1;
            double big = fabs(A[0][1]);
            if (fabs(A[0][2]) > big) { big = fabs(A[0][2]); p = 0; q = 2; }
            if (fabs(A[1][2]) > big) { big = fabs(A[1][2]); p = 1; q = 2; }
            if (big < 1e-12) break;
            double theta = (A[q][q] - A[p][p]) / (2.0 * A[p][q]);
            double t = (theta >= 0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
            double c = 1.0 / sqrt(t * t + 1.0), s = t * c;
            for (int k = 0; k < 3; k++) {
                double kp = A[k][p], kq = A[k][q];
                A[k][p] = c * kp - s * kq;
                A[k][q] = s * kp + c * kq;
            }
            for (int k = 0; k < 3; k++) {
                double pk = A[p][k], qk = A[q][k];
                A[p][k] = c * pk - s * qk;
                A[q][k] = s * pk + c * qk;
            }
            for (int k = 0; k < 3; k++) {
                double kp = V[k][p], kq = V[k][q];
                V[k][p] = c * kp - s * kq;
                V[k][q] = s * kp + c * kq;
            }
        }
        int order[3] = { 0, 1, 2 };
        std::sort(order, order + 3, [&](int a, int b) { return A[a][a] > A[b][b]; });
        for (int i = 0; i < 3; i++) {
            int k = order[i];
            vals[i] = A[k][k];
            vecs[i] = Vector3Normalize({ (float)V[0][k], (float)V[1][k], (float)V[2][k] });
        }
    }

    SuperFrustum BoxAround(const std::vector<Vector3>& P, float trim, float minSize) {
        Vector3 mean = { 0, 0, 0 };
        for (Vector3 p : P) mean = Vector3Add(mean, p);
        mean = Vector3Scale(mean, 1.0f / P.size());
        double cov[3][3] = {};
        for (Vector3 p : P) {
            Vector3 q = Vector3Subtract(p, mean);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++) cov[i][j] += (double)Coord(q, i) * Coord(q, j);
        }
        double vals[3];
        Vector3 axes[3];
        Eigen3(cov, vals, axes);
        Vector3 z = axes[0], x = axes[1];
        Vector3 y = Vector3Normalize(Vector3CrossProduct(z, x));
        x = Vector3CrossProduct(y, z);
        Vector3 dir[3] = { x, y, z };
        Vector3 size = { 0, 0, 0 }, centre = mean;
        std::vector<float> proj(P.size());
        for (int a = 0; a < 3; a++) {
            for (size_t i = 0; i < P.size(); i++) proj[i] = Vector3DotProduct(Vector3Subtract(P[i], mean), dir[a]);
            std::sort(proj.begin(), proj.end());
            size_t i0 = (size_t)(proj.size() * trim), i1 = std::min(proj.size() - 1, (size_t)(proj.size() * (1.0f - trim)));
            float lo = proj[i0], hi = proj[i1];
            Coord(size, a) = fmaxf(hi - lo, minSize);
            centre = Vector3Add(centre, Vector3Scale(dir[a], (lo + hi) * 0.5f));
        }
        SuperFrustum f;
        f.position = centre;
        f.rotation = FromAxes(x, y, z);
        f.size = size;
        return f;
    }

    // ---- pool ----------------------------------------------------------------------------------------------------------

    Pool::Pool(int workers) {
        for (int i = 0; i < workers; i++) m_Threads.emplace_back([this] { Work(); });
    }

    Pool::~Pool() {
        {
            std::lock_guard<std::mutex> l(m_Mutex);
            m_Quit = true;
        }
        m_Wake.notify_all();
        for (std::thread& t : m_Threads) t.join();
    }

    void Pool::For(int n, const std::function<void(int)>& f) {
        if (m_Threads.empty() || n <= 1) {
            for (int i = 0; i < n; i++) f(i);
            return;
        }
        {
            std::lock_guard<std::mutex> l(m_Mutex);
            m_Job = &f;
            m_Count = n;
            m_Next = 0;
            m_Finished = 0;
            m_Generation++;
        }
        m_Wake.notify_all();
        Drain();
        std::unique_lock<std::mutex> l(m_Mutex);
        m_Done.wait(l, [&] { return m_Finished == (int)m_Threads.size(); });
        m_Job = nullptr;
    }

    void Pool::ForEach(int n, const std::function<void(int)>& f) {
        const int run = 128;
        For((n + run - 1) / run, [&](int k) {
            for (int i = k * run; i < std::min(n, (k + 1) * run); i++) f(i);
        });
    }

    void Pool::Drain() {
        for (;;) {
            int i = m_Next.fetch_add(1);
            if (i >= m_Count) return;
            (*m_Job)(i);
        }
    }

    void Pool::Work() {
        int seen = 0;
        std::unique_lock<std::mutex> l(m_Mutex);
        for (;;) {
            m_Wake.wait(l, [&] { return m_Quit || m_Generation != seen; });
            if (m_Quit) return;
            seen = m_Generation;
            l.unlock();
            Drain();
            l.lock();
            if (++m_Finished == (int)m_Threads.size()) m_Done.notify_one();
        }
    }
}
