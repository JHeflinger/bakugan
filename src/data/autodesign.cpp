#include "autodesign.hpp"
#include "data/field.hpp"
#include "util/utils.hpp"
#include <raymath.h>
#include <algorithm>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <condition_variable>
#include <functional>
#include <memory>
#include <random>
#include <thread>
#include <unordered_map>

namespace Bakugan {
    namespace {
        using Rng = std::mt19937;
        using Clock = std::chrono::steady_clock;

        float Uniform(Rng& r, float a = 0.0f, float b = 1.0f) { return std::uniform_real_distribution<float>(a, b)(r); }
        float Normal(Rng& r, float s = 1.0f) { return std::normal_distribution<float>(0.0f, s)(r); }
        int Pick(Rng& r, int n) { return n <= 1 ? 0 : std::uniform_int_distribution<int>(0, n - 1)(r); }
        bool Chance(Rng& r, float p) { return Uniform(r) < p; }
        float Coord(Vector3 v, int i) { return (&v.x)[i]; }

        Vector3 RandomUnit(Rng& r) {
            for (;;) {
                Vector3 v = { Normal(r), Normal(r), Normal(r) };
                float l = Vector3Length(v);
                if (l > 1e-4f) return Vector3Scale(v, 1.0f / l);
            }
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

        // Rotation whose local x, y, z axes point along the given orthonormal, right handed directions.
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

        // Eigen decomposition of a symmetric 3x3 matrix (Jacobi), largest eigenvalue first.
        void Eigen3(const double in[3][3], double vals[3], Vector3 vecs[3]) {
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

        float BoxDistance(Vector3 p, const BoundingBox& b) {
            float dx = fmaxf(fmaxf(b.min.x - p.x, p.x - b.max.x), 0.0f);
            float dy = fmaxf(fmaxf(b.min.y - p.y, p.y - b.max.y), 0.0f);
            float dz = fmaxf(fmaxf(b.min.z - p.z, p.z - b.max.z), 0.0f);
            return sqrtf(dx * dx + dy * dy + dz * dz);
        }

        BoundingBox GrowBox(BoundingBox b, float m) {
            return { Vector3SubtractValue(b.min, m), Vector3AddValue(b.max, m) };
        }

        // An oriented box around a point cloud: local z along its longest principal direction, extents trimmed by
        // `trim` at each end.
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
                (&size.x)[a] = fmaxf(hi - lo, minSize);
                centre = Vector3Add(centre, Vector3Scale(dir[a], (lo + hi) * 0.5f));
            }
            SuperFrustum f;
            f.position = centre;
            f.rotation = FromAxes(x, y, z);
            f.size = size;
            return f;
        }

        // ---- a small thread pool: the score is thousands of independent field evaluations ------------------------------

        class Pool {
        public:
            explicit Pool(int workers) {
                for (int i = 0; i < workers; i++) m_Threads.emplace_back([this] { Work(); });
            }
            ~Pool() {
                {
                    std::lock_guard<std::mutex> l(m_Mutex);
                    m_Quit = true;
                }
                m_Wake.notify_all();
                for (std::thread& t : m_Threads) t.join();
            }
            Pool(const Pool&) = delete;
            Pool& operator=(const Pool&) = delete;

            // Calls f(0) .. f(n - 1), spread over the workers and the calling thread; returns when all are done.
            void For(int n, const std::function<void(int)>& f) {
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
        private:
            void Drain() {
                for (;;) {
                    int i = m_Next.fetch_add(1);
                    if (i >= m_Count) return;
                    (*m_Job)(i);
                }
            }
            void Work() {
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
        private:
            std::vector<std::thread> m_Threads;
            std::mutex m_Mutex;
            std::condition_variable m_Wake, m_Done;
            const std::function<void(int)>* m_Job = nullptr;
            int m_Count = 0;
            std::atomic<int> m_Next{ 0 };
            int m_Finished = 0;
            int m_Generation = 0;
            bool m_Quit = false;
        };

        // ---- inside / outside of the input mesh --------------------------------------------------------------------

        // Counts ray crossings along +x, +y and +z and takes the majority of the three parities, so a mesh with a few
        // holes or doubled faces still mostly answers right. Triangles are binned on a grid per axis.
        class InsideTest {
        public:
            InsideTest(const TriMesh& m, BoundingBox box) : m_Mesh(m) {
                size_t tris = m.indices.size() / 3;
                int n = std::clamp((int)sqrtf((float)tris / 4.0f), 4, 160);
                for (int a = 0; a < 3; a++) {
                    Axis& ax = m_Axes[a];
                    ax.a = a;
                    ax.u = (a + 1) % 3;
                    ax.v = (a + 2) % 3;
                    ax.n = n;
                    ax.u0 = Coord(box.min, ax.u);
                    ax.v0 = Coord(box.min, ax.v);
                    ax.du = fmaxf(Coord(box.max, ax.u) - ax.u0, 1e-6f) / n;
                    ax.dv = fmaxf(Coord(box.max, ax.v) - ax.v0, 1e-6f) / n;
                    ax.cells.assign((size_t)n * n, {});
                    for (size_t t = 0; t < tris; t++) {
                        float ulo = FLT_MAX, uhi = -FLT_MAX, vlo = FLT_MAX, vhi = -FLT_MAX;
                        for (int k = 0; k < 3; k++) {
                            Vector3 p = m.positions[m.indices[t * 3 + k]];
                            ulo = fminf(ulo, Coord(p, ax.u)); uhi = fmaxf(uhi, Coord(p, ax.u));
                            vlo = fminf(vlo, Coord(p, ax.v)); vhi = fmaxf(vhi, Coord(p, ax.v));
                        }
                        int i0 = std::clamp((int)floorf((ulo - ax.u0) / ax.du), 0, n - 1);
                        int i1 = std::clamp((int)floorf((uhi - ax.u0) / ax.du), 0, n - 1);
                        int j0 = std::clamp((int)floorf((vlo - ax.v0) / ax.dv), 0, n - 1);
                        int j1 = std::clamp((int)floorf((vhi - ax.v0) / ax.dv), 0, n - 1);
                        for (int j = j0; j <= j1; j++)
                            for (int i = i0; i <= i1; i++) ax.cells[(size_t)j * n + i].push_back((unsigned)t);
                    }
                }
            }

            bool Inside(Vector3 p) const {
                int votes = 0;
                for (const Axis& ax : m_Axes) votes += Crossings(ax, p) & 1;
                return votes >= 2;
            }
        private:
            struct Axis {
                int a = 0, u = 1, v = 2, n = 1;
                float u0 = 0, v0 = 0, du = 1, dv = 1;
                std::vector<std::vector<unsigned>> cells;
            };

            int Crossings(const Axis& ax, Vector3 p) const {
                // an odd offset keeps rays off shared edges and vertices of regular meshes
                float pu = Coord(p, ax.u) + ax.du * 1.37e-3f, pv = Coord(p, ax.v) + ax.dv * 2.71e-3f, pa = Coord(p, ax.a);
                int i = (int)floorf((pu - ax.u0) / ax.du), j = (int)floorf((pv - ax.v0) / ax.dv);
                if (i < 0 || j < 0 || i >= ax.n || j >= ax.n) return 0;
                int count = 0;
                for (unsigned t : ax.cells[(size_t)j * ax.n + i]) {
                    Vector3 A = m_Mesh.positions[m_Mesh.indices[t * 3]];
                    Vector3 B = m_Mesh.positions[m_Mesh.indices[t * 3 + 1]];
                    Vector3 C = m_Mesh.positions[m_Mesh.indices[t * 3 + 2]];
                    float au = Coord(A, ax.u), av = Coord(A, ax.v), bu = Coord(B, ax.u), bv = Coord(B, ax.v);
                    float cu = Coord(C, ax.u), cv = Coord(C, ax.v);
                    float e0 = (bu - au) * (pv - av) - (bv - av) * (pu - au);   // weight of C
                    float e1 = (cu - bu) * (pv - bv) - (cv - bv) * (pu - bu);   // weight of A
                    float e2 = (au - cu) * (pv - cv) - (av - cv) * (pu - cu);   // weight of B
                    bool pos = e0 >= 0 && e1 >= 0 && e2 >= 0, neg = e0 <= 0 && e1 <= 0 && e2 <= 0;
                    if (!pos && !neg) continue;
                    float area = e0 + e1 + e2;
                    if (fabsf(area) < 1e-20f) continue;
                    float h = (Coord(A, ax.a) * e1 + Coord(B, ax.a) * e2 + Coord(C, ax.a) * e0) / area;
                    if (h > pa) count++;
                }
                return count;
            }
        private:
            const TriMesh& m_Mesh;
            Axis m_Axes[3];
        };

        // ---- the mesh's signed distance, anywhere -------------------------------------------------------------------

        // Exact Euclidean distance transform (Felzenszwalb & Huttenlocher) of one row of squared distances, in place.
        void Edt1D(std::vector<double>& f, int n, std::vector<double>& d, std::vector<int>& v, std::vector<double>& z) {
            auto cross = [&](int q, int r) { return ((f[q] + (double)q * q) - (f[r] + (double)r * r)) / (2.0 * q - 2.0 * r); };
            int k = 0;
            v[0] = 0;
            z[0] = -1e30;
            z[1] = 1e30;
            for (int q = 1; q < n; q++) {
                double s = cross(q, v[k]);
                while (s <= z[k]) s = cross(q, v[--k]);
                v[++k] = q;
                z[k] = s;
                z[k + 1] = 1e30;
            }
            k = 0;
            for (int q = 0; q < n; q++) {
                while (z[k + 1] < q) k++;
                double dq = q - v[k];
                d[q] = dq * dq + f[v[k]];
            }
            for (int q = 0; q < n; q++) f[q] = d[q];
        }

        // Squared distance (in cells) from every grid point to the nearest point with `feature` set.
        std::vector<double> Edt3D(const std::vector<char>& feature, const int n[3]) {
            std::vector<double> g(feature.size());
            for (size_t i = 0; i < g.size(); i++) g[i] = feature[i] ? 0.0 : 1e20;
            int big = std::max(n[0], std::max(n[1], n[2]));
            std::vector<double> f(big), d(big), z(big + 1);
            std::vector<int> v(big);
            auto at = [&](int x, int y, int zz) { return ((size_t)zz * n[1] + y) * n[0] + x; };
            for (int zz = 0; zz < n[2]; zz++)
                for (int y = 0; y < n[1]; y++) {
                    for (int x = 0; x < n[0]; x++) f[x] = g[at(x, y, zz)];
                    Edt1D(f, n[0], d, v, z);
                    for (int x = 0; x < n[0]; x++) g[at(x, y, zz)] = f[x];
                }
            for (int zz = 0; zz < n[2]; zz++)
                for (int x = 0; x < n[0]; x++) {
                    for (int y = 0; y < n[1]; y++) f[y] = g[at(x, y, zz)];
                    Edt1D(f, n[1], d, v, z);
                    for (int y = 0; y < n[1]; y++) g[at(x, y, zz)] = f[y];
                }
            for (int y = 0; y < n[1]; y++)
                for (int x = 0; x < n[0]; x++) {
                    for (int zz = 0; zz < n[2]; zz++) f[zz] = g[at(x, y, zz)];
                    Edt1D(f, n[2], d, v, z);
                    for (int zz = 0; zz < n[2]; zz++) g[at(x, y, zz)] = f[zz];
                }
            return g;
        }

        // The mesh's signed distance (negative inside) on a grid around it, trilinear in between and growing with the
        // distance to the grid outside it: how far design material is from the mesh, wherever it is.
        class MeshField {
        public:
            template <class InsideFn>
            void Build(BoundingBox box, float pad, int res, const InsideFn& inside, Pool& pool) {
                box = GrowBox(box, pad);
                Vector3 ext = Vector3Subtract(box.max, box.min);
                m_H = fmaxf(ext.x, fmaxf(ext.y, ext.z)) / res;
                m_Min = box.min;
                for (int a = 0; a < 3; a++) m_N[a] = std::max(2, (int)ceilf(Coord(ext, a) / m_H) + 1);
                size_t count = (size_t)m_N[0] * m_N[1] * m_N[2];
                std::vector<char> in(count, 0);
                int rows = m_N[1] * m_N[2];
                pool.For(rows, [&](int r) {
                    int y = r % m_N[1], z = r / m_N[1];
                    for (int x = 0; x < m_N[0]; x++)
                        in[((size_t)z * m_N[1] + y) * m_N[0] + x] = inside(Point(x, y, z));
                });
                std::vector<char> out(count);
                for (size_t i = 0; i < count; i++) out[i] = !in[i];
                std::vector<double> toIn = Edt3D(in, m_N), toOut = Edt3D(out, m_N);
                size_t filled = 0;
                for (char c : in) filled += c;
                m_Volume = (double)filled * m_H * m_H * m_H;
                m_D.resize(count);
                for (size_t i = 0; i < count; i++)
                    m_D[i] = in[i] ? -((float)sqrt(toOut[i]) - 0.5f) * m_H : ((float)sqrt(toIn[i]) - 0.5f) * m_H;
            }

            float Distance(Vector3 p) const {
                Vector3 g = Vector3Scale(Vector3Subtract(p, m_Min), 1.0f / m_H);
                float c[3], out = 0.0f;
                for (int a = 0; a < 3; a++) {
                    float v = Coord(g, a), hi = (float)(m_N[a] - 1);
                    float cl = std::clamp(v, 0.0f, hi);
                    out += (v - cl) * (v - cl);
                    c[a] = cl;
                }
                int i[3];
                float t[3];
                for (int a = 0; a < 3; a++) {
                    i[a] = std::min((int)c[a], m_N[a] - 2);
                    t[a] = c[a] - i[a];
                }
                auto at = [&](int x, int y, int z) { return m_D[((size_t)z * m_N[1] + y) * m_N[0] + x]; };
                float v = 0.0f;
                for (int k = 0; k < 8; k++) {
                    int dx = k & 1, dy = (k >> 1) & 1, dz = (k >> 2) & 1;
                    float w = (dx ? t[0] : 1.0f - t[0]) * (dy ? t[1] : 1.0f - t[1]) * (dz ? t[2] : 1.0f - t[2]);
                    v += w * at(i[0] + dx, i[1] + dy, i[2] + dz);
                }
                return v + sqrtf(out) * m_H;
            }
            double Volume() const { return m_Volume; }
        private:
            Vector3 Point(int x, int y, int z) const { return { m_Min.x + x * m_H, m_Min.y + y * m_H, m_Min.z + z * m_H }; }
            Vector3 m_Min = { 0, 0, 0 };
            float m_H = 1.0f;
            int m_N[3] = { 2, 2, 2 };
            std::vector<float> m_D;
            double m_Volume = 0.0;
        };

        // ---- what the score compares against -------------------------------------------------------------------------

        struct Target {
            float L = 1.0f;                  // diagonal of the mesh's box: the length scale of everything
            BoundingBox box = {};
            Vector3 centre = { 0, 0, 0 };    // of the mesh's volume
            std::vector<Vector3> vol;        // volume samples (uniform in the box and near the surface)
            std::vector<char> volIn;         // ... inside the mesh
            std::vector<Vector3> surf;       // on the mesh surface (area weighted)
            std::vector<Vector3> inside;     // a pool of points inside the mesh
        };

        // Fixed unit samples for the closed pose (common random numbers: every design sees the same points, so score
        // differences come from the designs, not from sampling noise).
        struct Samples {
            std::vector<Vector3> openUnit;       // in [0, 1]^3, mapped into the open design's box (material off the mesh, collisions)
            std::vector<Vector3> closedUnit;     // in [0, 1]^3, mapped into the closed shape's box (volume, centroid, collisions)
            std::vector<Vector3> sphereUnit;     // in [-1.35, 1.35]^3, around the sphere (overlap)
            std::vector<char> sphereIn;          // ... inside the unit sphere
            std::vector<Vector3> shellDirs;      // on the unit sphere (its surface)
            std::vector<Vector3> pairUnit;       // in [0, 1]^3, mapped into where two bodies' boxes meet (motion checks)
        };

        // The bodies of a design as fields, for scoring. Distances are capped: anything farther than the cap counts
        // the same, which lets whole bodies be skipped by their boxes.
        struct Probe {
            std::vector<BodyField> fields;
            std::vector<int> group;              // group index of each body
            std::vector<int> id;                 // group id
            std::vector<BoundingBox> box;        // design pose

            void Build(const Design& d, float margin) {
                for (int g : d.Bodies()) {
                    BodyField f = BodyField::Body(d, d.groups[g].id);
                    if (f.Empty()) continue;
                    BoundingBox b = d.BodyBounds(d.groups[g].id);
                    float blend = 0.0f;
                    for (int c : d.ChildGroups(d.groups[g].id)) blend = fmaxf(blend, d.groups[c].blend);
                    box.push_back(GrowBox(b, fmaxf(blend, d.groups[g].blend) + margin));
                    fields.push_back(std::move(f));
                    group.push_back(g);
                    id.push_back(d.groups[g].id);
                }
            }

            int Index(int groupId) const {
                for (int i = 0; i < (int)id.size(); i++)
                    if (id[i] == groupId) return i;
                return -1;
            }

            // Distance to the union of the bodies, each moved by xf (indexed like the groups; empty: design pose).
            float Distance(Vector3 p, const std::vector<Rigid>* xf, float cap, int* who = nullptr) const {
                float best = cap;
                int w = -1;
                for (int i = 0; i < (int)fields.size(); i++) {
                    Vector3 q = xf ? (*xf)[group[i]].ApplyInverse(p) : p;
                    float bd = BoxDistance(q, box[i]);
                    if (bd > 0.0f && bd >= best) continue;
                    float d = fields[i].Distance(q);
                    if (d < best) { best = d; w = i; }
                }
                if (who) *who = w >= 0 ? group[w] : -1;
                return best;
            }

            // The nearest two bodies: d1 is the union's distance, d2 the next body's (negative where two bodies overlap).
            void Distance2(Vector3 p, const std::vector<Rigid>* xf, float cap, float* d1, float* d2, int* w1, int* w2) const {
                float b1 = cap, b2 = cap;
                int i1 = -1, i2 = -1;
                for (int i = 0; i < (int)fields.size(); i++) {
                    Vector3 q = xf ? (*xf)[group[i]].ApplyInverse(p) : p;
                    float bd = BoxDistance(q, box[i]);
                    if (bd > 0.0f && bd >= b2) continue;
                    float d = fields[i].Distance(q);
                    if (d < b1) { b2 = b1; i2 = i1; b1 = d; i1 = i; }
                    else if (d < b2) { b2 = d; i2 = i; }
                }
                *d1 = b1;
                *d2 = b2;
                *w1 = i1 >= 0 ? group[i1] : -1;
                *w2 = i2 >= 0 ? group[i2] : -1;
            }
        };

        bool IsIdentity(const std::vector<Rigid>& xf) {
            for (const Rigid& r : xf)
                if (fabsf(r.rotation.w) < 0.999999f || Vector3Length(r.position) > 1e-5f) return false;
            return true;
        }

        // A spot the last score found wrong, and the body (group index) nearest to it there.
        struct Spot {
            Vector3 p = { 0, 0, 0 };
            int body = -1;
        };

        struct Diagnostics {
            std::vector<Spot> missing;   // open pose: inside the mesh, outside the design
            std::vector<Spot> excess;    // open pose: inside the design, outside the mesh
            std::vector<Spot> holes;     // closed pose: inside the sphere, outside the closed shape
            std::vector<Spot> bumps;     // closed pose: inside the closed shape, outside the sphere
            std::vector<Spot> clashes;   // inside two bodies (body = the first), world position at that moment ...
            std::vector<int> clashOther; // ... the second body (group index)
            std::vector<float> clashT;   // ... the moment (0 open, 1 closed)
            std::vector<char> clashOpening;  // ... of the opening motion (else closing)
        };

        // A hinge's timing windows kept inside the motion, each a tenth of it at least (a hinge that snaps over in an
        // instant could swing through another body between two checked moments).
        void ClampTiming(Bakugan::Hinge& h) {
            auto window = [](float& a, float& b) {
                float w = std::clamp(b - a, 0.1f, 1.0f);
                a = std::clamp(a, 0.0f, 1.0f - w);
                b = a + w;
            };
            window(h.t0, h.t1);
            if (h.ownOpening) window(h.o0, h.o1);
        }

        // A spot where two bodies overlap at a checked moment of the motion.
        struct MotionClash {
            Vector3 p;                   // world position then
            int a = -1, b = -1;          // group indices of the two bodies
            float t = 0.0f;
            char opening = 0;
        };

        // What a structural change touched, so it can be tuned a little before it is judged.
        struct Focus {
            int shape = -1;              // shape id
            int hinge = -1;              // hinge id
            int group = -1;              // group id (its blend)
            int part = -1;               // only one kind of parameter (PartOf), or all of them (-1)
        };

        // Kinds of parameters that are moved together in a descent step: a shape's position, size, rotation or
        // profile; a hinge's pivot, axis or angle; a blend.
        int PartOf(int dof) {
            static const int parts[] = { 0, 1, 2, 3, 3, 3, 3, 4, 5, 6, 7, 8 };   // in Dof order
            return parts[dof];
        }

        struct State {
            Design design;
            AutoScore score;
            Diagnostics diag;
        };

        // ---- silhouettes -----------------------------------------------------------------------------------------------

        // One orthographic view: looking along `dir`, with `u` to the right and `v` up in the picture.
        struct SilView {
            Vector3 dir = { 0, 0, -1 }, u = { 1, 0, 0 }, v = { 0, 1, 0 };
        };

        // Every view frames the same square, `half` wide either side of `centre` (2·half across, `res` pixels).
        struct SilFrame {
            Vector3 centre = { 0, 0, 0 };
            float half = 1.0f;
            int res = 64;
            std::vector<SilView> views;
            float Pixel() const { return 2.0f * half / res; }
            // where a world point lands in a view, in pixels (x right, y down)
            Vector2 ToPixel(const SilView& w, Vector3 p) const {
                Vector3 q = Vector3Subtract(p, centre);
                return { (Vector3DotProduct(q, w.u) / half + 1.0f) * 0.5f * res, (1.0f - Vector3DotProduct(q, w.v) / half) * 0.5f * res };
            }
        };

        // `n` directions spread evenly over the upper half of a sphere (a Fibonacci spiral): from steep top views to
        // nearly level side views. Silhouettes from opposite directions are mirror images, so a half covers them all.
        std::vector<SilView> MakeViews(int n) {
            std::vector<SilView> out;
            for (int i = 0; i < n; i++) {
                float y = 1.0f - (i + 0.5f) / n;             // height of the eye above the centre, 1 (top) .. 0 (level)
                float r = sqrtf(fmaxf(0.0f, 1.0f - y * y)), phi = i * 2.39996323f + 0.6f;
                SilView w;
                Vector3 eye = { r * cosf(phi), y, r * sinf(phi) };
                w.dir = Vector3Negate(eye);
                Vector3 up = fabsf(w.dir.y) < 0.99f ? Vector3{ 0, 1, 0 } : Vector3{ 0, 0, -1 };
                w.u = Vector3Normalize(Vector3CrossProduct(w.dir, up));
                w.v = Vector3CrossProduct(w.u, w.dir);
                out.push_back(w);
            }
            return out;
        }

        // The mesh's outline in one view: triangles filled at 4x4 samples per pixel, then each pixel's covered share.
        std::vector<float> RasterMesh(const TriMesh& m, const SilFrame& fr, const SilView& w) {
            const int ss = 4, n = fr.res * ss;
            std::vector<unsigned char> hit((size_t)n * n, 0);
            std::vector<Vector2> pts(m.positions.size());
            for (size_t i = 0; i < pts.size(); i++) pts[i] = Vector2Scale(fr.ToPixel(w, m.positions[i]), (float)ss);
            for (size_t t = 0; t + 2 < m.indices.size(); t += 3) {
                Vector2 a = pts[m.indices[t]], b = pts[m.indices[t + 1]], c = pts[m.indices[t + 2]];
                float area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
                if (fabsf(area) < 1e-12f) continue;
                int x0 = std::max(0, (int)floorf(fminf(a.x, fminf(b.x, c.x)))), x1 = std::min(n - 1, (int)ceilf(fmaxf(a.x, fmaxf(b.x, c.x))));
                int y0 = std::max(0, (int)floorf(fminf(a.y, fminf(b.y, c.y)))), y1 = std::min(n - 1, (int)ceilf(fmaxf(a.y, fmaxf(b.y, c.y))));
                for (int y = y0; y <= y1; y++)
                    for (int x = x0; x <= x1; x++) {
                        float px = x + 0.5f, py = y + 0.5f;
                        float e0 = (b.x - a.x) * (py - a.y) - (b.y - a.y) * (px - a.x);
                        float e1 = (c.x - b.x) * (py - b.y) - (c.y - b.y) * (px - b.x);
                        float e2 = (a.x - c.x) * (py - c.y) - (a.y - c.y) * (px - c.x);
                        if ((e0 >= 0 && e1 >= 0 && e2 >= 0) || (e0 <= 0 && e1 <= 0 && e2 <= 0)) hit[(size_t)y * n + x] = 1;
                    }
            }
            std::vector<float> out((size_t)fr.res * fr.res, 0.0f);
            for (int y = 0; y < n; y++)
                for (int x = 0; x < n; x++)
                    if (hit[(size_t)y * n + x]) out[(size_t)(y / ss) * fr.res + x / ss] += 1.0f / (ss * ss);
            return out;
        }

        // One body's outline in one view, rows [y0, y1): a ray per pixel along the view direction, sphere traced
        // through the body's field (in the body's own frame) inside its box. A pixel's coverage comes from the
        // nearest the ray gets to the body, or the deepest into it, over a pixel's width: 0 when it passes half a
        // pixel outside, 1 when it goes half a pixel inside, linear between. That is roughly the share of the pixel
        // the body covers, and it changes smoothly as the body moves, which the descent needs (hard 0/1 pixels
        // would not change at all under a small step).
        void RenderBody(const BodyField& f, const BoundingBox& box, const Rigid& xf, const SilFrame& fr, const SilView& w,
                        int y0, int y1, unsigned char* out) {
            const float tau = 0.5f * fr.Pixel();
            const float far = 4.0f * fr.half + Vector3Length(Vector3Subtract(box.max, box.min));
            // the view in the body's frame
            Quaternion inv = { -xf.rotation.x, -xf.rotation.y, -xf.rotation.z, xf.rotation.w };
            Vector3 dir = Vector3RotateByQuaternion(w.dir, inv), u = Vector3RotateByQuaternion(w.u, inv), v = Vector3RotateByQuaternion(w.v, inv);
            Vector3 base = xf.ApplyInverse(Vector3Subtract(fr.centre, Vector3Scale(w.dir, far)));
            for (int y = y0; y < y1; y++)
                for (int x = 0; x < fr.res; x++) {
                    float sx = ((x + 0.5f) / fr.res * 2.0f - 1.0f) * fr.half, sy = (1.0f - (y + 0.5f) / fr.res * 2.0f) * fr.half;
                    Vector3 o = Vector3Add(base, Vector3Add(Vector3Scale(u, sx), Vector3Scale(v, sy)));
                    // the ray's stretch inside the box (slabs)
                    float tn = 0.0f, tf = 2.0f * far;
                    bool miss = false;
                    for (int a = 0; a < 3 && !miss; a++) {
                        float oa = Coord(o, a), da = Coord(dir, a), lo = Coord(box.min, a) - tau, hi = Coord(box.max, a) + tau;
                        if (fabsf(da) < 1e-9f) { miss = oa < lo || oa > hi; continue; }
                        float t0 = (lo - oa) / da, t1 = (hi - oa) / da;
                        if (t0 > t1) std::swap(t0, t1);
                        tn = fmaxf(tn, t0);
                        tf = fminf(tf, t1);
                        miss = tn > tf;
                    }
                    unsigned char cover = 0;
                    if (!miss) {
                        float t = tn, dmin = FLT_MAX;
                        for (int k = 0; k < 64 && t <= tf; k++) {
                            float d = f.Distance(Vector3Add(o, Vector3Scale(dir, t)));
                            dmin = fminf(dmin, d);
                            if (d < -tau) break;                       // well inside: fully covered
                            t += d > tau ? 0.9f * d : 0.5f * tau;     // sphere tracing, small steps near the surface
                        }
                        float c = std::clamp(0.5f - dmin / (2.0f * tau), 0.0f, 1.0f);
                        cover = (unsigned char)lroundf(c * 255.0f);
                    }
                    out[(size_t)y * fr.res + x] = cover;
                }
        }

        // FNV-1a, for the silhouette cache's keys.
        struct Hasher {
            unsigned long long h = 1469598103934665603ull;
            void Bytes(const void* p, size_t n) {
                const unsigned char* b = (const unsigned char*)p;
                for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
            }
            template <class T> void Add(const T& v) { Bytes(&v, sizeof(T)); }
        };

        // Everything a body's field depends on: its groups' operations and blends, nested as they are, and its shapes'
        // operations and values (hinge hardware is off while searching, so it never adds to a body).
        void HashGroup(const Design& d, int groupIndex, Hasher& hs) {
            const Group& g = d.groups[groupIndex];
            hs.Add((int)g.op);
            hs.Add(g.blend);
            for (int s : d.DirectShapes(g.id)) {
                const Shape& sh = d.shapes[s];
                const SuperFrustum& f = sh.sf;
                hs.Add((int)sh.op);
                hs.Add(f.position); hs.Add(f.rotation); hs.Add(f.size);
                hs.Add(f.rounding); hs.Add(f.taper); hs.Add(f.bulge); hs.Add(f.dilation); hs.Add(f.onion);
            }
            for (int c : d.ChildGroups(g.id)) {
                hs.Add(-7);              // marks nesting
                HashGroup(d, c, hs);
            }
        }

        // Rendered body outlines (every view, one pose), kept by what was rendered: the body's contents and where the
        // pose puts it. A descent step usually changes one body, so the others are found here instead of traced again.
        class SilCache {
        public:
            void Limit(size_t entryBytes) { m_Cap = std::max<size_t>(32, (size_t)(96u << 20) / std::max<size_t>(1, entryBytes)); }
            std::vector<unsigned char>* Find(unsigned long long key) {
                auto it = m_Map.find(key);
                if (it == m_Map.end()) return nullptr;
                it->second.used = ++m_Tick;
                return &it->second.pixels;
            }
            std::vector<unsigned char>* Add(unsigned long long key, size_t bytes) {
                if (m_Map.size() >= m_Cap) {
                    // drop the least recently used quarter
                    std::vector<std::pair<unsigned long long, unsigned long long>> age;
                    for (const auto& [k, e] : m_Map) age.push_back({ e.used, k });
                    std::sort(age.begin(), age.end());
                    for (size_t i = 0; i < age.size() / 4; i++) m_Map.erase(age[i].second);
                }
                Entry& e = m_Map[key];
                e.used = ++m_Tick;
                e.pixels.assign(bytes, 0);
                return &e.pixels;
            }
        private:
            struct Entry {
                std::vector<unsigned char> pixels;
                unsigned long long used = 0;
            };
            std::unordered_map<unsigned long long, Entry> m_Map;
            unsigned long long m_Tick = 0;
            size_t m_Cap = 256;
        };

        // ---- the grammar's rewrites -----------------------------------------------------------------------------

        enum class Rule : unsigned char {
            AddShape, AddCut, RemoveShape, ToggleOp, MoveShape,           // shapes and material
            AddBody, SplitBody, RemoveBody, MergeBody,                     // bodies
            AddFeature, DissolveFeature, ToggleFeature,                    // blended features
            AddHinge, RemoveHinge, Rehang, TurnHinge,                      // hinges
            RemoveStray, CarveClash,                                       // data-driven repairs
            Retime, OwnOpening,                                            // hinge timing
            Count
        };

        // One rewrite, with everything random about it already decided, so it can be applied to the design it was
        // sampled for, alone or together with other rewrites. Elements are named by id (stable while a design is
        // edited), never by index.
        struct Rewrite {
            Rule rule = Rule::AddShape;
            int shape = -1;              // the shape it acts on
            int group = -1;              // the body or feature it acts on, or that receives a shape
            int other = -1;              // a second body (a hinge's parent, a shape's new body)
            int hinge = -1;              // the hinge it acts on
            std::vector<int> shapes;     // shapes it groups (AddFeature)
            bool hasShape = false;       // `sf` is a new shape, or `shape`'s new values
            SuperFrustum sf;
            bool hasHinge = false;       // a new hinge's (or `hinge`'s new) pivot, axis and turn
            bool hasTiming = false;      // ... and its timing: when it turns while closing (t0..t1), and while opening
            float t0 = 0.0f, t1 = 1.0f;  //     (o0..o1, of the opening motion) if it has its own opening timing
            bool ownOpening = false;
            float o0 = 0.0f, o1 = 1.0f;
            Vector3 pivot = { 0, 0, 0 }, axis = { 0, 1, 0 };
            float angle = 0.0f;          // closing turn (end - start)
            float value = 0.0f;          // a new feature's blend
            std::vector<long long> owns; // elements it changes or removes ...
            std::vector<long long> uses; // ... and elements it only needs to still be there
            float gain = 0.0f;           // how much it improved the objective in its lookahead
        };

        long long Key(char kind, int id) { return ((long long)kind << 32) | (unsigned)id; }
        const long long TreeKey = ((long long)'T' << 32);   // the hinge tree's shape: one change to it per round

        // Two rewrites conflict when one changes something the other changes or needs.
        bool Conflict(const Rewrite& a, const Rewrite& b) {
            auto hits = [](const std::vector<long long>& x, const std::vector<long long>& y) {
                for (long long k : x)
                    if (std::find(y.begin(), y.end(), k) != y.end()) return true;
                return false;
            };
            return hits(a.owns, b.owns) || hits(a.owns, b.uses) || hits(b.owns, a.uses);
        }

        // ---- continuous parameters ------------------------------------------------------------------------------

        // A continuous degree of freedom. Values are moved in units of DofScale (the paper's grammar-specific scaling:
        // one unit is a small, comparable change for every kind), sizes on a log scale, rotations as small turns.
        enum class Dof : unsigned char { Pos, LogSize, Rot, Rounding, Taper, Bulge, Dilation, Pivot, Axis, Angle, Blend, Timing };
        struct Param {
            Dof dof;
            int id;                      // shape, hinge or group id
            int axis;                    // 0..2 for the vector ones
        };

        // ---- the search ------------------------------------------------------------------------------------------------

        class Search {
        public:
            Search(const TriMesh& mesh, const AutoParams& p, JobControl& c, std::shared_ptr<AutoLive> live)
                : m_Mesh(mesh), m_P(p), m_C(c), m_Live(std::move(live)), m_Rng(p.seed),
                  m_Pool(std::max(0, (int)std::thread::hardware_concurrency() - 1)) {
                m_P.maxBodies = std::max(1, m_P.maxBodies);
                m_P.minBodies = std::clamp(m_P.minBodies, 1, m_P.maxBodies);
                m_P.rewritesPerRound = std::max(1, m_P.rewritesPerRound);
                m_P.stepsPerRound = std::max(0, m_P.stepsPerRound);
                m_P.stepSize = std::max(1e-3f, m_P.stepSize);
            }

            AutoResult Run(const Design& start);
        private:
            bool Prepare();
            void MakeSamples();
            Design Seed();
            Design FromCurrent(const Design& start);
            AutoScore Evaluate(const Design& d, Diagnostics* diag);
            void ScoreMotion(const Design& d, const Probe& pr, double designVolume, AutoScore* sc, std::vector<MotionClash>* spots);
            void ScoreSilhouettes(const Design& d, const Probe& pr, const std::vector<Rigid>& open,
                                  const std::vector<Rigid>& closed, float offOpen, float offClosed, AutoScore* sc);
            void ViewsImage(std::vector<unsigned char>* rgba, int* w, int* h) const;
            float Objective(const AutoScore& s) const { return s.base + m_Ramp * m_P.collisionWeight * s.collision; }
            void Consider(const Design& d, const AutoScore& s);     // keeps it if it is the best so far
            bool Sanitize(Design& d) const;
            void Project(Design& d) const;
            void Finalize(Design& d) const;
            void Publish(bool force);

            // descent
            float DofScale(Dof k) const;
            std::vector<Param> Params(const Design& d, const Focus* only) const;
            std::vector<Param> AllParams(const Design& d, const Focus* only) const;
            void Step(Design& d, const std::vector<Param>& ps, const std::vector<float>& x) const;
            Focus RandomBlock(const Design& d);
            bool DescentStep(State& cur, const std::vector<Param>& ps, float lr);
            void Lookahead(State& cand, const Focus& f);

            // rewrites
            bool Sample(const Design& d, const Diagnostics& dg, Rewrite* r);
            bool Apply(Design& d, const Rewrite& r, Focus* f) const;
            bool ApplyRule(Design& d, const Rewrite& r, Focus* f) const;
            bool Timing() const { return m_P.allowTiming && m_P.motionChecks > 0 && m_P.collisionWeight > 0.0f; }
            void Remember(Rewrite& r, const Design& d, const Focus& f) const;   // the lookahead's values into the rewrite
            int RewriteRound(State& cur);

            // grammar helpers
            SuperFrustum NewShape(Vector3 at, bool cut);
            bool PlaceSpot(const Design& d, const Diagnostics& dg, bool forCut, Vector3* at, int* bodyGroup, bool* sphere = nullptr);
            bool FitRegion(const Design& d, const Diagnostics& dg, bool cut, SuperFrustum* f, int* bodyGroup);
            int NearestBody(const Design& d, Vector3 p, int except = -1) const;   // group index
            Vector3 BodyHint(const Design& d, int bodyId) const;                   // a point inside the body
            Vector3 PivotBetween(const Design& d, int bodyA, int bodyB, Vector3 a, Vector3 b) const;
            void ChooseHinge(Vector3 pivot, Vector3 childPoint, Vector3* axis, float* angle);
            void SetupHinge(Design& d, int index, Vector3 pivot, Vector3 axis, float angle, bool fresh) const;
            int AttachHinge(Design& d, int parent, int child, Vector3 pivot, Vector3 childPoint);
            void RemoveBody(Design& d, int bodyId) const;
            bool MergeBody(Design& d, int bodyId) const;
            int MaterialShapes(const Design& d, int bodyId) const;
            std::vector<int> Roots(const Design& d) const;                         // group ids of bodies on no hinge
        private:
            const TriMesh& m_Mesh;
            AutoParams m_P;
            JobControl& m_C;
            std::shared_ptr<AutoLive> m_Live;
            Rng m_Rng;
            Pool m_Pool;
            Target m_T;
            Samples m_S;
            float m_Tau = 1.0f;              // occupancy softness (mm)
            std::unique_ptr<InsideTest> m_Inside;
            MeshField m_MeshField;
            // silhouettes
            SilFrame m_Frame;
            std::vector<std::vector<float>> m_Target;          // the mesh's outline in each view (pixel coverage 0..1)
            SilCache m_Cache;
            std::vector<unsigned char> m_LastOpen, m_LastClosed;   // the design scored last: its outlines, every view
            std::vector<unsigned char> m_BestOpen, m_BestClosed;   // ... and the best design's
            // the box the search stays in (shared by every design, so a start design outside it is never stuck)
            float m_Reach = 0.0f;            // how far shapes and pivots may be from the mesh's centre
            float m_MinSize = 0.0f, m_MaxSize = 0.0f, m_MaxDilation = 0.0f, m_MaxBlend = 0.0f;
            AutoScore m_Score;               // of the design rewrites are sampled for (its sphere)
            std::unordered_map<int, bool> m_Hardware;   // hinge id -> hardware flag of a design we started from
            // the run
            float m_Ramp = 0.1f;             // share of the collision weight in force (it grows over the run)
            float m_Lr = 0.5f;               // descent step (parameter scale units)
            float m_SpsaRms = 0.0f;          // running RMS of SPSA slopes: steps are measured against it
            State m_Best;
            int m_Accepted = 0, m_Rounds = 0;
            Clock::time_point m_LastPublish;
            int m_Evaluations = 0;
        };

        // ---- preparation -----------------------------------------------------------------------------------------------

        bool Search::Prepare() {
            m_C.SetStage("Sampling the mesh");
            const TriMesh& m = m_Mesh;
            size_t tris = m.indices.size() / 3;
            if (tris == 0) return false;
            Vector3 lo = { FLT_MAX, FLT_MAX, FLT_MAX }, hi = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
            for (unsigned i : m.indices) { lo = Vector3Min(lo, m.positions[i]); hi = Vector3Max(hi, m.positions[i]); }
            m_T.box = { lo, hi };
            m_T.L = fmaxf(Vector3Distance(lo, hi), 1e-3f);
            float L = m_T.L;
            m_Tau = 0.008f * L;
            m_Reach = 1.2f * L;
            m_MinSize = 0.008f * L;
            m_MaxSize = 1.2f * L;
            m_MaxDilation = 0.2f * L;
            m_MaxBlend = 0.04f * L;

            // area weighted surface sampling
            std::vector<double> cdf(tris);
            std::vector<Vector3> normals(tris);
            double total = 0.0;
            for (size_t t = 0; t < tris; t++) {
                Vector3 a = m.positions[m.indices[t * 3]], b = m.positions[m.indices[t * 3 + 1]], c = m.positions[m.indices[t * 3 + 2]];
                Vector3 n = Vector3CrossProduct(Vector3Subtract(b, a), Vector3Subtract(c, a));
                float area = Vector3Length(n) * 0.5f;
                normals[t] = area > 0.0f ? Vector3Scale(n, 0.5f / area) : Vector3{ 0, 1, 0 };
                total += area;
                cdf[t] = total;
            }
            if (total <= 0.0) return false;
            auto surfacePoint = [&](Vector3* normal) {
                double r = Uniform(m_Rng) * total;
                size_t t = std::min((size_t)(std::lower_bound(cdf.begin(), cdf.end(), r) - cdf.begin()), tris - 1);
                float u = Uniform(m_Rng), v = Uniform(m_Rng);
                if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
                Vector3 a = m.positions[m.indices[t * 3]], b = m.positions[m.indices[t * 3 + 1]], c = m.positions[m.indices[t * 3 + 2]];
                if (normal) *normal = normals[t];
                return Vector3Add(a, Vector3Add(Vector3Scale(Vector3Subtract(b, a), u), Vector3Scale(Vector3Subtract(c, a), v)));
            };

            const int surfCount[3] = { 600, 1200, 2400 };
            const int volCount[3] = { 2000, 4000, 8000 };
            int q = std::clamp(m_P.quality, 0, 2);
            for (int i = 0; i < surfCount[q]; i++) m_T.surf.push_back(surfacePoint(nullptr));
            BoundingBox pad = GrowBox(m_T.box, 0.06f * L);
            Vector3 ext = Vector3Subtract(pad.max, pad.min);
            for (int i = 0; i < volCount[q]; i++) {
                if (i % 2 == 0) {
                    m_T.vol.push_back({ pad.min.x + Uniform(m_Rng) * ext.x, pad.min.y + Uniform(m_Rng) * ext.y, pad.min.z + Uniform(m_Rng) * ext.z });
                } else {
                    // near the surface, where the fit is decided
                    Vector3 n;
                    Vector3 s = surfacePoint(&n);
                    m_T.vol.push_back(Vector3Add(s, Vector3Scale(n, Normal(m_Rng, 0.03f * L))));
                }
            }
            if (m_C.Cancelled()) return false;

            m_C.SetStage("Finding the inside of the mesh");
            m_Inside = std::make_unique<InsideTest>(m, m_T.box);
            const InsideTest& test = *m_Inside;
            m_T.volIn.assign(m_T.vol.size(), 0);
            const int chunk = 128;
            int chunks = ((int)m_T.vol.size() + chunk - 1) / chunk;
            m_Pool.For(chunks, [&](int k) {
                for (int i = k * chunk; i < std::min((int)m_T.vol.size(), (k + 1) * chunk); i++) m_T.volIn[i] = test.Inside(m_T.vol[i]);
            });
            for (size_t i = 0; i < m_T.vol.size(); i++)
                if (m_T.volIn[i]) m_T.inside.push_back(m_T.vol[i]);
            if (m_C.Cancelled()) return false;
            m_C.SetStage("Measuring distances to the mesh");
            const int res[3] = { 48, 72, 96 };
            m_MeshField.Build(m_T.box, 0.2f * L, res[q], [&](Vector3 p) { return test.Inside(p); }, m_Pool);

            if (m_P.loss == AutoLoss::Silhouettes) {
                // every view frames the same square around the mesh's box, a little wider than the mesh can ever
                // reach (it is all within half the box's diagonal of the centre); material beyond the frame is caught
                // by the off-frame share instead
                if (m_C.Cancelled()) return false;
                m_C.SetStage("Drawing the mesh's outlines");
                m_Frame.centre = Vector3Scale(Vector3Add(lo, hi), 0.5f);
                m_Frame.half = 0.55f * L;
                m_Frame.res = std::clamp(m_P.viewResolution, 16, 256);
                m_Frame.views = MakeViews(std::clamp(m_P.views, 1, 24));
                m_Target.assign(m_Frame.views.size(), {});
                m_Pool.For((int)m_Frame.views.size(), [&](int v) { m_Target[v] = RasterMesh(m, m_Frame, m_Frame.views[v]); });
                m_Cache.Limit(m_Frame.views.size() * m_Frame.res * m_Frame.res);
            }
            // more inside points from the box, for seeding and placing new shapes
            for (int round = 0; round < 6 && m_T.inside.size() < 2000; round++) {
                std::vector<Vector3> cand(4000);
                for (Vector3& p : cand)
                    p = { m_T.box.min.x + Uniform(m_Rng) * (hi.x - lo.x), m_T.box.min.y + Uniform(m_Rng) * (hi.y - lo.y),
                          m_T.box.min.z + Uniform(m_Rng) * (hi.z - lo.z) };
                std::vector<char> in(cand.size(), 0);
                m_Pool.For((int)cand.size() / chunk, [&](int k) {
                    for (int i = k * chunk; i < (k + 1) * chunk; i++) in[i] = test.Inside(cand[i]);
                });
                for (size_t i = 0; i < cand.size(); i++)
                    if (in[i]) m_T.inside.push_back(cand[i]);
            }
            if (m_T.inside.size() < 20) return false;
            Vector3 c = { 0, 0, 0 };
            for (Vector3 p : m_T.inside) c = Vector3Add(c, p);
            m_T.centre = Vector3Scale(c, 1.0f / m_T.inside.size());
            return true;
        }

        void Search::MakeSamples() {
            const int closedCount[3] = { 1000, 2000, 4000 };
            const int sphereCount[3] = { 1200, 2500, 5000 };
            const int shellCount[3] = { 200, 400, 800 };
            const int pairCount[3] = { 64, 128, 256 };
            int q = std::clamp(m_P.quality, 0, 2);
            Rng r(m_P.seed * 7919u + 17u);
            for (int i = 0; i < closedCount[q]; i++) m_S.closedUnit.push_back({ Uniform(r), Uniform(r), Uniform(r) });
            for (int i = 0; i < closedCount[q]; i++) m_S.openUnit.push_back({ Uniform(r), Uniform(r), Uniform(r) });
            for (int i = 0; i < sphereCount[q]; i++) {
                Vector3 u = { Uniform(r, -1.35f, 1.35f), Uniform(r, -1.35f, 1.35f), Uniform(r, -1.35f, 1.35f) };
                m_S.sphereUnit.push_back(u);
                m_S.sphereIn.push_back(Vector3Length(u) <= 1.0f);
            }
            for (int i = 0; i < pairCount[q]; i++) m_S.pairUnit.push_back({ Uniform(r), Uniform(r), Uniform(r) });
            // a Fibonacci sphere: even coverage of the surface
            int n = shellCount[q];
            for (int i = 0; i < n; i++) {
                float y = 1.0f - 2.0f * (i + 0.5f) / n, rad = sqrtf(fmaxf(0.0f, 1.0f - y * y)), phi = i * 2.39996323f;
                m_S.shellDirs.push_back({ rad * cosf(phi), y, rad * sinf(phi) });
            }
        }

        // ---- the score -------------------------------------------------------------------------------------------------

        // One number for both poses, lower is better. Below is the Volume loss; with Silhouettes (ScoreSilhouettes),
        // fit and sphere are replaced by the outline comparisons, and the volume and surface samples are only measured
        // when a rewrite round needs their spots. Collisions, costs and the sphere's size and centre are the same.

        //   fit       open pose against the mesh: the mesh volume the design misses and the design volume outside the
        //             mesh (sampled wherever the design is), both relative to the mesh's volume, how far out that extra
        //             material reaches, and how far the mesh surface is from the design surface
        //   sphere    closed pose against a sphere of its own volume around its own centroid: the same overlap and
        //             surface measures, and how far closed material reaches outside the sphere
        //   total     (1 - bias) * fit + bias * sphere + collisions * weight + costs
        // Collisions are bodies overlapping each other in the open pose and in the closed pose, and with motion checks
        // at evenly spaced moments of the motion between them (ScoreMotion).
        // Costs keep it simple and buildable: per shape and group, pivots off their bodies, bodies on no hinge.
        AutoScore Search::Evaluate(const Design& d, Diagnostics* diag) {
            m_Evaluations++;
            AutoScore sc;
            const float L = m_T.L, tau = m_Tau;
            Probe pr;
            pr.Build(d, 0.002f * L);
            if (pr.fields.empty()) return sc;

            std::vector<Rigid> open = d.Transforms(0.0f, false), closed = d.Transforms(1.0f, false);
            const std::vector<Rigid>* openXf = IsIdentity(open) ? nullptr : &open;

            // the boxes of the open design and of the closed shape
            auto poseBox = [&](const std::vector<Rigid>& xf, Vector3* lo, Vector3* hi) {
                *lo = { FLT_MAX, FLT_MAX, FLT_MAX };
                *hi = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
                for (size_t i = 0; i < pr.fields.size(); i++) {
                    const BoundingBox& b = pr.box[i];
                    for (int k = 0; k < 8; k++) {
                        Vector3 c = { (k & 1) ? b.max.x : b.min.x, (k & 2) ? b.max.y : b.min.y, (k & 4) ? b.max.z : b.min.z };
                        Vector3 w = xf[pr.group[i]].Apply(c);
                        *lo = Vector3Min(*lo, w);
                        *hi = Vector3Max(*hi, w);
                    }
                }
            };
            Vector3 olo, ohi, clo, chi;
            poseBox(open, &olo, &ohi);
            poseBox(closed, &clo, &chi);
            Vector3 oext = Vector3Subtract(ohi, olo), cext = Vector3Subtract(chi, clo);
            auto inBox = [](Vector3 lo, Vector3 ext, Vector3 u) { return Vector3{ lo.x + u.x * ext.x, lo.y + u.y * ext.y, lo.z + u.z * ext.z }; };

            const float occCap = 6.0f * tau;     // the soft occupancy is flat past this
            const float fitCap = 0.08f * L;
            // scored by silhouettes, the volume and surface samples only find the spots a rewrite round aims at; the
            // samples over the open and closed designs are always needed (collisions, the sphere's size and centre)
            const bool silhouettes = m_P.loss == AutoLoss::Silhouettes;
            const bool useVolume = !silhouettes || diag;
            int nV = useVolume ? (int)m_T.vol.size() : 0, nS = useVolume ? (int)m_T.surf.size() : 0;
            int nO = (int)m_S.openUnit.size(), nC = (int)m_S.closedUnit.size();
            int oBase = nV + nS, cBase = oBase + nO, n1 = cBase + nC;
            std::vector<float> dist(n1), dist2(n1, occCap), meshD(nO, 0.0f);
            std::vector<int> who(n1, -1), who2(n1, -1);
            const int chunk = 128;
            m_Pool.For((n1 + chunk - 1) / chunk, [&](int k) {
                for (int i = k * chunk; i < std::min(n1, (k + 1) * chunk); i++) {
                    if (i < nV) {
                        dist[i] = pr.Distance(m_T.vol[i], openXf, occCap, &who[i]);
                    } else if (i < oBase) {
                        dist[i] = pr.Distance(m_T.surf[i - nV], openXf, fitCap, &who[i]);
                    } else if (i < cBase) {
                        Vector3 p = inBox(olo, oext, m_S.openUnit[i - oBase]);
                        pr.Distance2(p, openXf, occCap, &dist[i], &dist2[i], &who[i], &who2[i]);
                        meshD[i - oBase] = m_MeshField.Distance(p);
                    } else {
                        Vector3 p = inBox(clo, cext, m_S.closedUnit[i - cBase]);
                        pr.Distance2(p, &closed, occCap, &dist[i], &dist2[i], &who[i], &who2[i]);
                    }
                }
            });
            auto occupancy = [&](float dd) { return 1.0f / (1.0f + expf(dd / tau)); };
            // overlap of two bodies: only penetration deeper than about tau counts, so bodies may touch
            auto clash = [&](float dd) { return occupancy(dd + tau); };

            // open pose against the mesh, on samples around the mesh
            double inter = 0.0, uni = 0.0, nIn = 0.0;
            std::vector<int> missing;
            std::vector<Spot> excess;
            for (int i = 0; i < nV; i++) {
                float a = occupancy(dist[i]);
                if (m_T.volIn[i]) { inter += a; uni += 1.0; nIn += 1.0; if (dist[i] > 0.0f) missing.push_back(i); }
                else { uni += a; if (dist[i] < 0.0f) excess.push_back({ m_T.vol[i], who[i] }); }
            }
            sc.fitIoU = uni > 0.0 ? (float)(inter / uni) : 0.0f;
            sc.fitCover = nIn > 0 ? (float)(inter / nIn) : 0.0f;
            double gap = 0.0;
            for (int i = nV; i < oBase; i++) gap += fminf(fabsf(dist[i]), fitCap);
            sc.fitGap = nS > 0 ? (float)(gap / nS / fitCap) : 0.0f;

            // open pose, on samples over the design's own volume: material off the mesh, and bodies overlapping
            std::vector<int> clashes;
            double designVolume = 0.0;                   // the design's material (mm^3), from the open samples
            float offOpen = 0.0f, offClosed = 0.0f;      // share of the material outside the views' frame
            auto outsideFrame = [&](Vector3 p) { return Vector3Distance(p, m_Frame.centre) > m_Frame.half; };
            {
                double mass = 0.0, off = 0.0, far = 0.0, overlap = 0.0, depth = 0.0, unseen = 0.0;
                for (int k = 0; k < nO; k++) {
                    int i = oBase + k;
                    float a = occupancy(dist[i]);
                    mass += a;
                    off += a * occupancy(-meshD[k]);
                    far += a * fmaxf(0.0f, meshD[k]);
                    if (dist2[i] < occCap) overlap += clash(dist2[i]);
                    depth += fminf(fmaxf(0.0f, -dist2[i]), occCap);
                    if (dist[i] < 0.0f && meshD[k] > 0.0f) excess.push_back({ inBox(olo, oext, m_S.openUnit[k]), who[i] });
                    if (dist2[i] < -tau) clashes.push_back(i);
                    if (silhouettes && outsideFrame(inBox(olo, oext, m_S.openUnit[k]))) unseen += a;
                }
                if (mass > 1e-6) {
                    offOpen = (float)(unseen / mass);
                    double sampleVolume = (double)oext.x * oext.y * oext.z / nO;
                    designVolume = mass * sampleVolume;
                    sc.fitOutside = (float)(off / mass);
                    sc.fitExcess = (float)(off * sampleVolume / fmax(m_MeshField.Volume(), 1e-9));
                    sc.fitSpill = (float)(far / mass / (0.1 * L));
                    sc.clashOpen = (float)(overlap / mass);
                    sc.collision += 0.5f * sc.clashOpen + 0.5f * (float)(depth / mass / occCap);
                }
            }
            // missing volume and extra volume, both as shares of the mesh's volume, weigh the same: a blob twice the
            // mesh's size costs as much as covering none of it
            sc.fit = 0.35f * (1.0f - sc.fitCover) + 0.35f * fminf(sc.fitExcess, 2.0f) + 0.15f * sc.fitGap + 0.15f * fminf(sc.fitSpill, 2.0f);

            // closed pose: volume and centroid, then the sphere it should be; bodies overlapping
            double mass = 0.0, overlapC = 0.0, depthC = 0.0, unseenC = 0.0;
            Vector3 centroid = { 0, 0, 0 };
            for (int k = 0; k < nC; k++) {
                int i = cBase + k;
                float a = occupancy(dist[i]);
                Vector3 p = inBox(clo, cext, m_S.closedUnit[k]);
                mass += a;
                centroid = Vector3Add(centroid, Vector3Scale(p, a));
                if (dist2[i] < occCap) overlapC += clash(dist2[i]);
                depthC += fminf(fmaxf(0.0f, -dist2[i]), occCap);
                if (dist2[i] < -tau) clashes.push_back(i);
                if (silhouettes && outsideFrame(p)) unseenC += a;
            }
            if (mass > 1e-6) {
                offClosed = (float)(unseenC / mass);
                sc.clashClosed = (float)(overlapC / mass);
                sc.collision += 0.5f * sc.clashClosed + 0.5f * (float)(depthC / mass / occCap);
            }
            std::vector<MotionClash> motionClashes;
            if (m_P.motionChecks > 0 && designVolume > 0.0) ScoreMotion(d, pr, designVolume, &sc, diag ? &motionClashes : nullptr);
            double boxVolume = (double)cext.x * cext.y * cext.z;
            double volume = nC > 0 ? mass / nC * boxVolume : 0.0;
            if (mass < 1e-3 || volume <= 0.0) {
                sc.sphere = 1.0f;
            } else {
                sc.centre = Vector3Scale(centroid, (float)(1.0 / mass));
                sc.radius = m_P.sphereDiameter > 0.0f ? m_P.sphereDiameter * 0.5f : (float)cbrt(3.0 * volume / (4.0 * PI));
            }
            if (sc.radius > 0.0f && useVolume) {
                float R = fmaxf(sc.radius, 1e-3f);
                const float shellCap = 0.3f * R;
                int nP = (int)m_S.sphereUnit.size(), nH = (int)m_S.shellDirs.size(), n2 = nP + nH;
                std::vector<float> d2(n2);
                std::vector<int> w2(n2, -1);
                m_Pool.For((n2 + chunk - 1) / chunk, [&](int k) {
                    for (int i = k * chunk; i < std::min(n2, (k + 1) * chunk); i++) {
                        if (i < nP) d2[i] = pr.Distance(Vector3Add(sc.centre, Vector3Scale(m_S.sphereUnit[i], R)), &closed, occCap, &w2[i]);
                        else d2[i] = pr.Distance(Vector3Add(sc.centre, Vector3Scale(m_S.shellDirs[i - nP], R)), &closed, shellCap, &w2[i]);
                    }
                });
                double si = 0.0, su = 0.0, sg = 0.0;
                std::vector<int> holes, bumps;
                for (int i = 0; i < nP; i++) {
                    float a = occupancy(d2[i]);
                    if (m_S.sphereIn[i]) { si += a; su += 1.0; if (d2[i] > 0.0f) holes.push_back(i); }
                    else { su += a; if (d2[i] < 0.0f) bumps.push_back(i); }
                }
                for (int i = nP; i < n2; i++) sg += fminf(fabsf(d2[i]), shellCap);
                // material outside the sphere, by how far out it is: a small part swung away from the ball barely
                // dents the overlap, but should still be pulled in
                double spill = 0.0;
                for (int k = 0; k < nC; k++) {
                    Vector3 p = inBox(clo, cext, m_S.closedUnit[k]);
                    spill += occupancy(dist[cBase + k]) * fmaxf(0.0f, Vector3Distance(p, sc.centre) - R) / R;
                }
                sc.sphereSpill = (float)(spill / mass);
                sc.sphereIoU = su > 0.0 ? (float)(si / su) : 0.0f;
                sc.sphereGap = nH > 0 ? (float)(sg / nH / shellCap) : 0.0f;
                sc.sphere = fminf(0.75f * (1.0f - sc.sphereIoU) + 0.25f * sc.sphereGap + 0.5f * sc.sphereSpill, 1.5f);
                if (diag) {
                    auto take = [&](const std::vector<int>& from, std::vector<Spot>* out) {
                        out->clear();
                        for (int k = 0; k < 24 && !from.empty(); k++) {
                            int i = from[Pick(m_Rng, (int)from.size())];
                            out->push_back({ Vector3Add(sc.centre, Vector3Scale(m_S.sphereUnit[i], R)), w2[i] });
                        }
                    };
                    take(holes, &diag->holes);
                    take(bumps, &diag->bumps);
                }
            }
            if (diag) {
                diag->missing.clear();
                for (int k = 0; k < 24 && !missing.empty(); k++) {
                    int i = missing[Pick(m_Rng, (int)missing.size())];
                    diag->missing.push_back({ m_T.vol[i], who[i] });
                }
                diag->excess.clear();
                for (int k = 0; k < 24 && !excess.empty(); k++) diag->excess.push_back(excess[Pick(m_Rng, (int)excess.size())]);
                diag->clashes.clear();
                diag->clashOther.clear();
                diag->clashT.clear();
                diag->clashOpening.clear();
                for (int k = 0; k < 24 && !clashes.empty(); k++) {
                    int i = clashes[Pick(m_Rng, (int)clashes.size())];
                    bool isClosed = i >= cBase;
                    Vector3 p = isClosed ? inBox(clo, cext, m_S.closedUnit[i - cBase]) : inBox(olo, oext, m_S.openUnit[i - oBase]);
                    diag->clashes.push_back({ p, who[i] });
                    diag->clashOther.push_back(who2[i]);
                    diag->clashT.push_back(isClosed ? 1.0f : 0.0f);
                    diag->clashOpening.push_back(0);
                }
                for (int k = 0; k < 16 && !motionClashes.empty(); k++) {
                    const MotionClash& mc = motionClashes[Pick(m_Rng, (int)motionClashes.size())];
                    diag->clashes.push_back({ mc.p, mc.a });
                    diag->clashOther.push_back(mc.b);
                    diag->clashT.push_back(mc.t);
                    diag->clashOpening.push_back(mc.opening);
                }
            }

            if (silhouettes) ScoreSilhouettes(d, pr, open, closed, offOpen, offClosed, &sc);

            // buildability and simplicity
            int features = 0;
            for (const Group& g : d.groups) features += g.parent >= 0;
            int bodies = (int)d.Bodies().size();
            float penalty = m_P.simplicity * 0.012f * ((float)d.shapes.size() + 0.5f * features + 0.5f * bodies);
            penalty += 0.1f * (float)std::max(0, (int)Roots(d).size() - 1);      // every body should hang on the hinge tree
            const float slack = 0.015f * L;
            for (const Bakugan::Hinge& h : d.hinges) {
                // the pivot has to sit on (or in) both bodies it joins
                int a = pr.Index(h.parent), b = pr.Index(h.child);
                float dp = a >= 0 ? pr.fields[a].Distance(h.pivot) : L, dc = b >= 0 ? pr.fields[b].Distance(h.pivot) : L;
                penalty += 3.0f * (fmaxf(0.0f, dp - slack) + fmaxf(0.0f, dc - slack)) / L;
            }
            sc.penalty = penalty;
            float w = std::clamp(m_P.sphereBias, 0.0f, 1.0f);
            sc.base = (1.0f - w) * sc.fit + w * sc.sphere + sc.coveragePenalty + penalty;
            sc.total = sc.base + m_P.collisionWeight * sc.collision;
            sc.valid = true;
            return sc;
        }

        // Bodies overlapping on the way: at motionChecks evenly spaced moments strictly between open and closed (of the
        // closing motion, and of the opening one too when a hinge has its own opening timing), eased as the editor
        // animates it. Two bodies can only overlap where their boxes do, so each moment only samples the meeting box
        // of every pair of bodies whose world boxes meet (a fixed set of points, the same for every design, so
        // scores compare without sampling noise). A point counts as overlapping as the endpoints' samples do: inside
        // both bodies by more than about tau, softly. A moment scores like an endpoint pose (half the share of the
        // design's volume inside two bodies, half how deep, capped) and the moments' mean adds to the collisions, so
        // the whole motion between weighs as much as one end pose, however many moments are checked.
        void Search::ScoreMotion(const Design& d, const Probe& pr, double designVolume, AutoScore* sc, std::vector<MotionClash>* spots) {
            const int n = std::max(0, m_P.motionChecks), nf = (int)pr.fields.size();
            if (n == 0 || nf < 2) return;
            const float tau = m_Tau, occCap = 6.0f * tau;
            auto occupancy = [&](float dd) { return 1.0f / (1.0f + expf(dd / tau)); };

            struct Moment {
                float t;
                bool opening;
                std::vector<Rigid> xf;
                double overlap = 0.0, depth = 0.0;
            };
            std::vector<Moment> moments;
            const int motions = d.HasOwnOpening() ? 2 : 1;
            for (int m = 0; m < motions; m++)
                for (int k = 1; k <= n; k++) {
                    float t = (float)k / (n + 1);
                    moments.push_back({ t, m == 1, d.Transforms(t, true, m == 1 ? Motion::Opening : Motion::Closing) });
                }

            // the pairs of bodies whose boxes meet, per moment
            struct Task {
                int moment, a, b;
                Vector3 lo, ext;
                double overlap = 0.0, depth = 0.0;
                std::vector<Vector3> deep;    // a few points well inside both
            };
            std::vector<Task> tasks;
            std::vector<Vector3> wlo(nf), whi(nf);
            for (int m = 0; m < (int)moments.size(); m++) {
                const std::vector<Rigid>& xf = moments[m].xf;
                for (int i = 0; i < nf; i++) {
                    const BoundingBox& b = pr.box[i];
                    wlo[i] = { FLT_MAX, FLT_MAX, FLT_MAX };
                    whi[i] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
                    for (int c = 0; c < 8; c++) {
                        Vector3 q = { (c & 1) ? b.max.x : b.min.x, (c & 2) ? b.max.y : b.min.y, (c & 4) ? b.max.z : b.min.z };
                        Vector3 w = xf[pr.group[i]].Apply(q);
                        wlo[i] = Vector3Min(wlo[i], w);
                        whi[i] = Vector3Max(whi[i], w);
                    }
                }
                for (int i = 0; i < nf; i++)
                    for (int j = i + 1; j < nf; j++) {
                        Vector3 lo = Vector3Max(wlo[i], wlo[j]), hi = Vector3Min(whi[i], whi[j]);
                        if (lo.x >= hi.x || lo.y >= hi.y || lo.z >= hi.z) continue;
                        Task t;
                        t.moment = m;
                        t.a = i;
                        t.b = j;
                        t.lo = lo;
                        t.ext = Vector3Subtract(hi, lo);
                        tasks.push_back(std::move(t));
                    }
            }
            auto inside = [](const BoundingBox& b, Vector3 q) {
                return q.x >= b.min.x && q.x <= b.max.x && q.y >= b.min.y && q.y <= b.max.y && q.z >= b.min.z && q.z <= b.max.z;
            };
            m_Pool.For((int)tasks.size(), [&](int k) {
                Task& t = tasks[k];
                const Rigid& xa = moments[t.moment].xf[pr.group[t.a]];
                const Rigid& xb = moments[t.moment].xf[pr.group[t.b]];
                double overlap = 0.0, depth = 0.0;
                for (const Vector3& u : m_S.pairUnit) {
                    Vector3 p = { t.lo.x + u.x * t.ext.x, t.lo.y + u.y * t.ext.y, t.lo.z + u.z * t.ext.z };
                    Vector3 qa = xa.ApplyInverse(p);
                    if (!inside(pr.box[t.a], qa)) continue;        // outside a body's box: outside the body
                    float da = pr.fields[t.a].Distance(qa);
                    if (da >= occCap) continue;
                    Vector3 qb = xb.ApplyInverse(p);
                    if (!inside(pr.box[t.b], qb)) continue;
                    float db = pr.fields[t.b].Distance(qb);
                    float dd = fmaxf(da, db);                       // inside both by dd (negative)
                    if (dd >= occCap) continue;
                    overlap += occupancy(dd + tau);
                    depth += fminf(fmaxf(0.0f, -dd), occCap);
                    if (dd < -tau && t.deep.size() < 4) t.deep.push_back(p);
                }
                double vol = (double)t.ext.x * t.ext.y * t.ext.z / m_S.pairUnit.size();
                t.overlap = overlap * vol;
                t.depth = depth * vol;
            });
            for (const Task& t : tasks) {
                moments[t.moment].overlap += t.overlap;
                moments[t.moment].depth += t.depth;
                if (spots)
                    for (const Vector3& p : t.deep)
                        spots->push_back({ p, pr.group[t.a], pr.group[t.b], moments[t.moment].t, (char)moments[t.moment].opening });
            }
            double sum = 0.0;
            for (const Moment& m : moments) {
                float share = (float)std::min(1.0, m.overlap / designVolume);
                sum += 0.5 * share + 0.5 * std::min(1.0, m.depth / designVolume / occCap);
                if (share > sc->clashMotion) {
                    sc->clashMotion = share;
                    sc->clashMotionT = m.t;
                    sc->clashMotionOpening = m.opening;
                }
            }
            sc->collision += (float)(sum / moments.size());
        }

        // The silhouette score. Every body's outline in every view, in the open and in the closed pose, comes from the
        // cache or is traced now (in parallel); a pose's outline is the union of its bodies' (each pixel the most any
        // body covers). Then, per view, against the target T (the mesh's outline for the open pose, the sphere's disc
        // for the closed one), with D the design's outline:
        //   covered = sum(min(T, D)) / sum(T)          the share of the target the design covers
        //   extra   = sum(max(0, D - T)) / sum(T)      the design's pixels where the target has none, by the same yardstick
        //   loss    = (1 - covered) + extra
        // fit and sphere are the mean losses over the views, plus twice the share of material outside the frame
        // (pixels there are never seen, so that material would otherwise be free). A view covered less than
        // minViewCoverage costs 10 for each whole unit it falls short (a 10% shortfall costs 1, more than the rest of
        // the score): steep enough that no view can be traded away, but still a slope, so a design short of it can
        // climb back (a flat "very bad" would leave the search with no direction). The mesh's views only count while
        // the goal leaves the mesh any weight, the sphere's only while it gives the sphere any.
        void Search::ScoreSilhouettes(const Design& d, const Probe& pr, const std::vector<Rigid>& open,
                                      const std::vector<Rigid>& closed, float offOpen, float offClosed, AutoScore* sc) {
            const SilFrame& fr = m_Frame;
            const int nv = (int)fr.views.size(), px = fr.res * fr.res;
            const size_t bytes = (size_t)nv * px;

            // each body's outlines in each pose: found in the cache, or queued to be traced
            struct Trace {
                std::vector<unsigned char>* out;
                int body;
                Rigid xf;
            };
            std::vector<Trace> traces;
            std::vector<std::vector<unsigned char>*> layers[2];
            for (int pose = 0; pose < 2; pose++) {
                const std::vector<Rigid>& xf = pose == 0 ? open : closed;
                for (size_t i = 0; i < pr.fields.size(); i++) {
                    const Rigid& r = xf[pr.group[i]];
                    Hasher hs;
                    HashGroup(d, pr.group[i], hs);
                    hs.Add(r.rotation);
                    hs.Add(r.position);
                    std::vector<unsigned char>* img = m_Cache.Find(hs.h);
                    if (!img) {
                        img = m_Cache.Add(hs.h, bytes);
                        traces.push_back({ img, (int)i, r });
                    }
                    layers[pose].push_back(img);
                }
            }
            const int rows = 8, blocks = (fr.res + rows - 1) / rows;
            m_Pool.For((int)traces.size() * nv * blocks, [&](int t) {
                const Trace& tr = traces[t / (nv * blocks)];
                int v = (t / blocks) % nv, b = t % blocks;
                RenderBody(pr.fields[tr.body], pr.box[tr.body], tr.xf, fr, fr.views[v], b * rows, std::min(fr.res, (b + 1) * rows),
                           tr.out->data() + (size_t)v * px);
            });
            for (int pose = 0; pose < 2; pose++) {
                std::vector<unsigned char>& out = pose == 0 ? m_LastOpen : m_LastClosed;
                out.assign(bytes, 0);
                for (const std::vector<unsigned char>* img : layers[pose])
                    for (size_t k = 0; k < bytes; k++) out[k] = std::max(out[k], (*img)[k]);
            }

            float w = std::clamp(m_P.sphereBias, 0.0f, 1.0f);
            const float minCover = std::clamp(m_P.minViewCoverage, 0.0f, 1.0f);
            float worst = 1.0f, penalty = 0.0f;
            auto compare = [&](const unsigned char* design, const auto& target, bool counts, float* cover, float* extra) {
                double tsum = 0.0, hit = 0.0, spill = 0.0;
                for (int k = 0; k < px; k++) {
                    float t = target(k), dd = design[k] * (1.0f / 255.0f);
                    tsum += t;
                    hit += fminf(t, dd);
                    spill += fmaxf(0.0f, dd - t);
                }
                *cover = tsum > 0.0 ? (float)(hit / tsum) : 0.0f;
                *extra = tsum > 0.0 ? (float)(spill / tsum) : (spill > 0.0 ? 1.0f : 0.0f);
                if (counts) {
                    worst = fminf(worst, *cover);
                    penalty += 10.0f * fmaxf(0.0f, minCover - *cover);
                }
            };

            // the open pose against the mesh's outlines
            double fit = 0.0, cover = 0.0, extra = 0.0;
            for (int v = 0; v < nv; v++) {
                float c, e;
                const std::vector<float>& target = m_Target[v];
                compare(m_LastOpen.data() + (size_t)v * px, [&](int k) { return target[k]; }, w < 1.0f, &c, &e);
                fit += (1.0f - c) + e;
                cover += c;
                extra += e;
            }
            sc->viewCover = (float)(cover / nv);
            sc->viewExtra = (float)(extra / nv);
            sc->fit = (float)(fit / nv) + 2.0f * offOpen;

            // the closed pose against the sphere: the same disc in every view, centred where the sphere's centre
            // lands in it (exact to a fraction of a pixel, so the score moves smoothly with the centre)
            if (sc->radius > 0.0f) {
                double sph = 0.0, scover = 0.0, sextra = 0.0;
                const float rpx = sc->radius / fr.Pixel();
                for (int v = 0; v < nv; v++) {
                    Vector2 c0 = fr.ToPixel(fr.views[v], sc->centre);
                    auto disc = [&](int k) {
                        float x = (k % fr.res) + 0.5f - c0.x, y = (k / fr.res) + 0.5f - c0.y;
                        return std::clamp(rpx - sqrtf(x * x + y * y) + 0.5f, 0.0f, 1.0f);
                    };
                    float c, e;
                    compare(m_LastClosed.data() + (size_t)v * px, disc, w > 0.0f, &c, &e);
                    sph += (1.0f - c) + e;
                    scover += c;
                    sextra += e;
                }
                sc->sphereViewCover = (float)(scover / nv);
                sc->sphereViewExtra = (float)(sextra / nv);
                sc->sphere = (float)(sph / nv) + 2.0f * offClosed;
            } else {
                sc->sphere = 2.0f;
                if (w > 0.0f) penalty += 10.0f * minCover * nv;
                worst = 0.0f;
            }
            sc->silhouettes = true;
            sc->worstView = worst;
            sc->coveragePenalty = penalty;
            sc->offFrame = offOpen + offClosed;
        }

        // The best design's views as one picture: a row of the open pose against the mesh, a row of the closed pose
        // against the sphere. Dark where both cover a pixel, red where only the target does (missing), blue where only
        // the design does (extra), white where neither.
        void Search::ViewsImage(std::vector<unsigned char>* rgba, int* w, int* h) const {
            const SilFrame& fr = m_Frame;
            const int nv = (int)fr.views.size(), px = fr.res * fr.res, gap = 3;
            if (m_BestOpen.size() != (size_t)nv * px) { *w = *h = 0; rgba->clear(); return; }
            *w = nv * fr.res + (nv + 1) * gap;
            *h = 2 * fr.res + 3 * gap;
            rgba->assign((size_t)*w * *h * 4, 0);
            for (size_t k = 0; k < rgba->size(); k += 4) { (*rgba)[k] = 222; (*rgba)[k + 1] = 225; (*rgba)[k + 2] = 232; (*rgba)[k + 3] = 255; }
            const float both[3] = { 70, 74, 86 }, miss[3] = { 235, 110, 95 }, extra[3] = { 90, 140, 235 };
            const float rpx = m_Best.score.radius / fr.Pixel();
            for (int row = 0; row < 2; row++)
                for (int v = 0; v < nv; v++) {
                    Vector2 c0 = fr.ToPixel(fr.views[v], m_Best.score.centre);
                    const unsigned char* design = (row == 0 ? m_BestOpen : m_BestClosed).data() + (size_t)v * px;
                    int ox = gap + v * (fr.res + gap), oy = gap + row * (fr.res + gap);
                    for (int k = 0; k < px; k++) {
                        int x = k % fr.res, y = k / fr.res;
                        float t;
                        if (row == 0) {
                            t = m_Target[v][k];
                        } else {
                            float dx = x + 0.5f - c0.x, dy = y + 0.5f - c0.y;
                            t = rpx > 0.0f ? std::clamp(rpx - sqrtf(dx * dx + dy * dy) + 0.5f, 0.0f, 1.0f) : 0.0f;
                        }
                        float dd = design[k] / 255.0f, ov = fminf(t, dd), ms = t - ov, ex = dd - ov, rest = 1.0f - fmaxf(t, dd);
                        unsigned char* o = &(*rgba)[((size_t)(oy + y) * *w + ox + x) * 4];
                        for (int c = 0; c < 3; c++)
                            o[c] = (unsigned char)std::clamp(lroundf(ov * both[c] + ms * miss[c] + ex * extra[c] + rest * 255.0f), 0L, 255L);
                    }
                }
        }

        // ---- helpers ---------------------------------------------------------------------------------------------------

        std::vector<int> Search::Roots(const Design& d) const {
            std::vector<int> out;
            for (int g : d.Bodies())
                if (d.ParentHinge(d.groups[g].id) < 0) out.push_back(d.groups[g].id);
            return out;
        }

        int Search::MaterialShapes(const Design& d, int bodyId) const {
            int n = 0;
            for (int s : d.GroupShapes(bodyId)) n += d.AddsMaterial(s);
            return n;
        }

        int Search::NearestBody(const Design& d, Vector3 p, int except) const {
            int best = -1;
            float bd = FLT_MAX;
            for (int g : d.Bodies()) {
                if (d.groups[g].id == except) continue;
                BodyField f = BodyField::Body(d, d.groups[g].id);
                if (f.Empty()) continue;
                float dist = f.Distance(p);
                if (dist < bd) { bd = dist; best = g; }
            }
            return best;
        }

        // The centre of the body's biggest material adding shape: inside the body unless a cut takes it.
        Vector3 Search::BodyHint(const Design& d, int bodyId) const {
            float big = -1.0f;
            Vector3 at = { 0, 0, 0 };
            for (int s : d.GroupShapes(bodyId)) {
                if (!d.AddsMaterial(s)) continue;
                const SuperFrustum& f = d.shapes[s].sf;
                float v = f.size.x * f.size.y * f.size.z + f.dilation * f.dilation * f.dilation * 4.0f;
                if (v > big) { big = v; at = f.position; }
            }
            return at;
        }

        // Where the two bodies meet between a (in A) and b (in B): the point on the segment as far from both.
        Vector3 Search::PivotBetween(const Design& d, int bodyA, int bodyB, Vector3 a, Vector3 b) const {
            BodyField fa = BodyField::Body(d, bodyA), fb = BodyField::Body(d, bodyB);
            if (fa.Empty() || fb.Empty()) return Vector3Lerp(a, b, 0.5f);
            float lo = 0.0f, hi = 1.0f;
            for (int k = 0; k < 14; k++) {
                float mid = (lo + hi) * 0.5f;
                Vector3 p = Vector3Lerp(a, b, mid);
                if (fa.Distance(p) < fb.Distance(p)) lo = mid;
                else hi = mid;
            }
            return Vector3Lerp(a, b, (lo + hi) * 0.5f);
        }

        // How a new hinge turns its child: half the time folding it towards the middle of the mesh (where the sphere
        // wants to be), about the axis that swings it there, most of the way; otherwise about a random axis across
        // the child's direction, by a random angle.
        void Search::ChooseHinge(Vector3 pivot, Vector3 childPoint, Vector3* axis, float* angle) {
            Vector3 out = Vector3Subtract(childPoint, pivot);
            if (Vector3Length(out) < 1e-4f) out = RandomUnit(m_Rng);
            out = Vector3Normalize(out);
            Vector3 in = Vector3Subtract(m_T.centre, pivot);
            Vector3 swing = Vector3CrossProduct(out, in);
            if (Chance(m_Rng, 0.5f) && Vector3Length(swing) > 1e-3f * Vector3Length(in)) {
                *axis = Vector3Normalize(swing);
                float cosang = Vector3DotProduct(out, Vector3Normalize(in));
                *angle = acosf(std::clamp(cosang, -1.0f, 1.0f)) * Uniform(m_Rng, 0.6f, 1.0f);
            } else {
                Vector3 r = RandomUnit(m_Rng);
                r = Vector3Subtract(r, Vector3Scale(out, Vector3DotProduct(r, out)));
                *axis = Vector3Length(r) > 1e-3f ? Vector3Normalize(r) : Utils::AnyPerpendicular(out);
                *angle = Uniform(m_Rng, -PI, PI);
            }
        }

        // Puts a pivot, an axis and a closing turn on hinge `index`. A fresh hinge also starts open at 0 and gets
        // hardware sized to the model (switched off: the score ignores it, Finalize decides).
        void Search::SetupHinge(Design& d, int index, Vector3 pivot, Vector3 axis, float angle, bool fresh) const {
            Bakugan::Hinge& hg = d.hinges[index];
            if (fresh) {
                const float L = m_T.L;
                hg.start = 0.0f;
                hg.hardware = false;
                hg.radius = std::clamp(0.03f * L, 1.6f, 4.0f);
                hg.pinRadius = hg.radius * 0.4f;
                hg.length = std::clamp(0.2f * L, 6.0f, 30.0f);
                hg.leafLength = std::clamp(0.07f * L, 2.0f, 10.0f);
                hg.leafThickness = std::clamp(0.025f * L, 1.2f, 3.0f);
                hg.knuckles = 3;
            }
            hg.pivot = pivot;
            hg.axis = Vector3Length(axis) > 1e-6f ? Vector3Normalize(axis) : Vector3{ 0, 1, 0 };
            hg.end = hg.start + angle;
        }

        int Search::AttachHinge(Design& d, int parent, int child, Vector3 pivot, Vector3 childPoint) {
            if (!d.CanHinge(parent, child)) return -1;
            int h = d.AddHinge(parent, child);
            Vector3 axis;
            float angle;
            ChooseHinge(pivot, childPoint, &axis, &angle);
            SetupHinge(d, h, pivot, axis, angle, true);
            return d.hinges[h].id;
        }

        // Takes a body out and keeps the hinge tree whole: its children hang from its parent instead (or, for a root,
        // from its first child, which becomes the root).
        void Search::RemoveBody(Design& d, int bodyId) const {
            int ph = d.ParentHinge(bodyId);
            int up = ph >= 0 ? d.hinges[ph].parent : -1;
            std::vector<int> kids;
            for (int h = 0; h < (int)d.hinges.size(); h++)
                if (d.hinges[h].parent == bodyId) kids.push_back(h);
            if (up >= 0) {
                for (int h : kids) d.hinges[h].parent = up;
            } else if (!kids.empty()) {
                int root = d.hinges[kids[0]].child;
                for (size_t k = 1; k < kids.size(); k++) d.hinges[kids[k]].parent = root;
            }
            d.RemoveGroup(d.GroupIndex(bodyId));
        }

        // Folds a hinged body back into its parent: its shapes and features join the parent, its children hang there.
        bool Search::MergeBody(Design& d, int bodyId) const {
            int ph = d.ParentHinge(bodyId);
            if (ph < 0) return false;
            int into = d.hinges[ph].parent;
            for (int s : d.DirectShapes(bodyId)) {
                d.MoveShape(s, into);
                if (d.shapes[s].op == Op::Union) d.shapes[s].color = d.groups[d.GroupIndex(into)].color;
            }
            std::vector<int> features;
            for (int g : d.ChildGroups(bodyId)) features.push_back(d.groups[g].id);
            for (int id : features) d.MoveGroup(d.GroupIndex(id), into);
            for (Bakugan::Hinge& h : d.hinges)
                if (h.parent == bodyId) h.parent = into;
            d.RemoveGroup(d.GroupIndex(bodyId));
            return true;
        }

        SuperFrustum Search::NewShape(Vector3 at, bool cut) {
            const float L = m_T.L;
            SuperFrustum f;
            f.position = at;
            f.rotation = RandomRotation(m_Rng);
            float lo = cut ? 0.04f : 0.05f, hi = cut ? 0.14f : 0.22f;
            f.size = { L * Uniform(m_Rng, lo, hi), L * Uniform(m_Rng, lo, hi), L * Uniform(m_Rng, lo, hi) };
            f.rounding = Uniform(m_Rng);
            f.taper = Chance(m_Rng, 0.7f) ? 1.0f : Uniform(m_Rng, 0.2f, 1.0f);
            return f;
        }

        // Somewhere worth adding material (or cutting it), in the design pose, and the body (group index) it belongs to.
        // The fit and the sphere each nominate spots, as often as they matter in the score.
        bool Search::PlaceSpot(const Design& d, const Diagnostics& dg, bool forCut, Vector3* at, int* body, bool* sphere) {
            float w = std::clamp(m_P.sphereBias, 0.0f, 1.0f);
            const std::vector<Spot>& fit = forCut ? dg.excess : dg.missing;
            const std::vector<Spot>& sph = forCut ? dg.bumps : dg.holes;
            bool useSphere = !sph.empty() && (fit.empty() || Chance(m_Rng, 0.15f + 0.7f * w));
            const std::vector<Spot>* from = useSphere ? &sph : &fit;
            if (sphere) *sphere = false;
            if (from->empty()) {
                if (forCut || m_T.inside.empty()) return false;
                *at = m_T.inside[Pick(m_Rng, (int)m_T.inside.size())];
                *body = NearestBody(d, *at);
                return *body >= 0;
            }
            if (sphere) *sphere = useSphere;
            Spot s = (*from)[Pick(m_Rng, (int)from->size())];
            int g = s.body;
            if (g < 0 || g >= (int)d.groups.size() || d.groups[g].parent >= 0) {
                // nothing near it: the nearest body in the pose the spot is in
                if (useSphere) {
                    std::vector<Rigid> closed = d.Transforms(1.0f, false);
                    float best = FLT_MAX;
                    for (int b : d.Bodies()) {
                        BodyField f = BodyField::Body(d, d.groups[b].id);
                        if (f.Empty()) continue;
                        float dist = f.Distance(closed[b].ApplyInverse(s.p));
                        if (dist < best) { best = dist; g = b; }
                    }
                } else {
                    g = NearestBody(d, s.p);
                }
            }
            if (g < 0) return false;
            std::vector<Rigid> xf = d.Transforms(useSphere ? 1.0f : 0.0f, false);
            *at = xf[g].ApplyInverse(s.p);
            *body = g;
            return true;
        }

        // A superfrustum for the region around a bad spot: points in a ball around it that are in the mesh but not the
        // design (or in the design but not the mesh, for a cut), or in the sphere but not the closed shape (or the other
        // way round), get an oriented box fitted to them, in the design frame of the body it goes into.
        bool Search::FitRegion(const Design& d, const Diagnostics& dg, bool cut, SuperFrustum* f, int* body) {
            Vector3 at;
            bool sphere = false;
            if (!PlaceSpot(d, dg, cut, &at, body, &sphere)) return false;
            const float L = m_T.L;
            if (sphere && m_Score.radius <= 0.0f) sphere = false;
            std::vector<Rigid> xf = d.Transforms(sphere ? 1.0f : 0.0f, false);
            Vector3 centre = xf[*body].Apply(at);       // the spot in its pose
            Probe pr;
            pr.Build(d, 0.002f * L);
            float rho = L * Uniform(m_Rng, 0.06f, 0.22f);
            std::vector<Vector3> keep;
            for (int i = 0; i < 400; i++) {
                Vector3 p;
                do p = { Uniform(m_Rng, -1, 1), Uniform(m_Rng, -1, 1), Uniform(m_Rng, -1, 1) };
                while (Vector3LengthSqr(p) > 1.0f);
                p = Vector3Add(centre, Vector3Scale(p, rho));
                int who = -1;
                float dd = pr.Distance(p, &xf, 0.1f * L, &who);
                bool want;
                if (sphere) {
                    bool inSphere = Vector3Distance(p, m_Score.centre) < m_Score.radius;
                    want = cut ? (!inSphere && dd < 0.0f && who == *body) : (inSphere && dd > 0.0f);
                } else {
                    bool inMesh = m_Inside->Inside(p);
                    want = cut ? (!inMesh && dd < 0.0f && who == *body) : (inMesh && dd > 0.0f);
                }
                if (want) keep.push_back(xf[*body].ApplyInverse(p));
            }
            if (keep.size() < 12) {
                *f = NewShape(at, cut);
                return true;
            }
            *f = BoxAround(keep, 0.04f, 0.015f * L);
            // smaller than the region: a rewrite should change the shape only a little (the paper's jump continuity),
            // and the descent grows it from inside the gap rather than shrinking it back out of the neighbours
            f->size = Vector3Scale(f->size, Uniform(m_Rng, 0.5f, 0.8f));
            f->rounding = Chance(m_Rng, 0.5f) ? Uniform(m_Rng, 0.4f, 1.0f) : Uniform(m_Rng);
            return true;
        }

        // ---- descent: the continuous half of SRD ---------------------------------------------------------------------

        // How far one unit of each kind of parameter moves it: comparable small changes for every kind, relative to
        // the mesh's size where it is a length.
        float Search::DofScale(Dof k) const {
            const float L = m_T.L;
            switch (k) {
                case Dof::Pos: return 0.01f * L;
                case Dof::LogSize: return 0.05f;         // 5% bigger or smaller
                case Dof::Rot: return 0.05f;             // radians
                case Dof::Rounding:
                case Dof::Taper:
                case Dof::Bulge: return 0.05f;
                case Dof::Dilation: return 0.004f * L;
                case Dof::Pivot: return 0.01f * L;
                case Dof::Axis: return 0.05f;            // radians
                case Dof::Angle: return 0.06f;           // radians
                case Dof::Blend: return 0.004f * L;
                case Dof::Timing: return 0.04f;          // share of the motion
            }
            return 1.0f;
        }

        // The continuous parameters of a design (or only those of the focused shape, hinge and group): per shape its
        // position, size, rotation, rounding, taper, bulge and dilation; per hinge its pivot, axis and closing angle;
        // per group of two or more shapes its blend. A shape's shell (onion) is left alone: it is all or nothing in
        // effect, so a small step of it carries no useful slope.
        std::vector<Param> Search::Params(const Design& d, const Focus* only) const {
            std::vector<Param> ps = AllParams(d, only);
            if (only && only->part >= 0)
                ps.erase(std::remove_if(ps.begin(), ps.end(), [&](const Param& p) { return PartOf((int)p.dof) != only->part; }), ps.end());
            return ps;
        }

        std::vector<Param> Search::AllParams(const Design& d, const Focus* only) const {
            std::vector<Param> ps;
            for (const Shape& sh : d.shapes) {
                if (only && sh.id != only->shape) continue;
                for (int a = 0; a < 3; a++) ps.push_back({ Dof::Pos, sh.id, a });
                for (int a = 0; a < 3; a++) ps.push_back({ Dof::LogSize, sh.id, a });
                for (int a = 0; a < 3; a++) ps.push_back({ Dof::Rot, sh.id, a });
                ps.push_back({ Dof::Rounding, sh.id, 0 });
                ps.push_back({ Dof::Taper, sh.id, 0 });
                ps.push_back({ Dof::Bulge, sh.id, 0 });
                ps.push_back({ Dof::Dilation, sh.id, 0 });
            }
            for (const Bakugan::Hinge& h : d.hinges) {
                if (only && h.id != only->hinge) continue;
                for (int a = 0; a < 3; a++) ps.push_back({ Dof::Pivot, h.id, a });
                for (int a = 0; a < 3; a++) ps.push_back({ Dof::Axis, h.id, a });
                ps.push_back({ Dof::Angle, h.id, 0 });
                if (Timing()) {
                    // closing window: where it sits, how long it is; the same for its own opening window
                    ps.push_back({ Dof::Timing, h.id, 0 });
                    ps.push_back({ Dof::Timing, h.id, 1 });
                    if (h.ownOpening) {
                        ps.push_back({ Dof::Timing, h.id, 2 });
                        ps.push_back({ Dof::Timing, h.id, 3 });
                    }
                }
            }
            for (const Group& g : d.groups) {
                if (only && g.id != only->group) continue;
                if (d.GroupShapes(g.id).size() >= 2) ps.push_back({ Dof::Blend, g.id, 0 });
            }
            return ps;
        }

        // Moves every parameter by x[i] of its units, then pulls values back into range. Rotations collect their three
        // components first and turn once, about the combined axis.
        void Search::Step(Design& d, const std::vector<Param>& ps, const std::vector<float>& x) const {
            std::unordered_map<int, Vector3> turnShape, turnAxis;
            for (size_t i = 0; i < ps.size(); i++) {
                const Param& p = ps[i];
                float v = x[i] * DofScale(p.dof);
                if (v == 0.0f) continue;
                if (p.dof == Dof::Rot) {
                    (&turnShape[p.id].x)[p.axis] += v;
                } else if (p.dof == Dof::Axis) {
                    (&turnAxis[p.id].x)[p.axis] += v;
                } else if (p.dof <= Dof::Dilation) {
                    int s = d.ShapeIndex(p.id);
                    if (s < 0) continue;
                    SuperFrustum& f = d.shapes[s].sf;
                    switch (p.dof) {
                        case Dof::Pos: (&f.position.x)[p.axis] += v; break;
                        case Dof::LogSize: (&f.size.x)[p.axis] *= expf(v); break;
                        case Dof::Rounding: f.rounding += v; break;
                        case Dof::Taper: f.taper += v; break;
                        case Dof::Bulge: f.bulge += v; break;
                        case Dof::Dilation: f.dilation += v; break;
                        default: break;
                    }
                } else if (p.dof == Dof::Timing) {
                    int h = d.HingeIndex(p.id);
                    if (h < 0) continue;
                    Bakugan::Hinge& hg = d.hinges[h];
                    float& a = p.axis < 2 ? hg.t0 : hg.o0;
                    float& b = p.axis < 2 ? hg.t1 : hg.o1;
                    if (p.axis % 2 == 0) { a += v; b += v; }               // slide the window
                    else { a -= 0.5f * v; b += 0.5f * v; }                 // stretch it about its middle
                } else if (p.dof == Dof::Blend) {
                    int g = d.GroupIndex(p.id);
                    if (g >= 0) d.groups[g].blend += v;
                } else {
                    int h = d.HingeIndex(p.id);
                    if (h < 0) continue;
                    if (p.dof == Dof::Pivot) (&d.hinges[h].pivot.x)[p.axis] += v;
                    else if (p.dof == Dof::Angle) d.hinges[h].end += v;
                }
            }
            auto turn = [](Vector3 r) {
                float a = Vector3Length(r);
                return a > 1e-9f ? QuaternionFromAxisAngle(Vector3Scale(r, 1.0f / a), a) : QuaternionIdentity();
            };
            for (const auto& [id, r] : turnShape) {
                int s = d.ShapeIndex(id);
                if (s >= 0) d.shapes[s].sf.rotation = QuaternionNormalize(QuaternionMultiply(turn(r), d.shapes[s].sf.rotation));
            }
            for (const auto& [id, r] : turnAxis) {
                int h = d.HingeIndex(id);
                if (h >= 0) d.hinges[h].axis = Vector3Normalize(Vector3RotateByQuaternion(d.hinges[h].axis, turn(r)));
            }
            Project(d);
        }

        // The parameters the next descent step moves: one element of the design (a shape, a hinge or a blended
        // group, each as likely as the next), and usually one kind of its parameters (a shape's position, size,
        // rotation or profile; a hinge's pivot, axis or angle), now and then all of them. In few dimensions a random
        // probe direction lines up with the gradient far more often.
        Focus Search::RandomBlock(const Design& d) {
            int blends = 0;
            for (const Group& g : d.groups) blends += d.GroupShapes(g.id).size() >= 2;
            int n = (int)d.shapes.size() + (int)d.hinges.size() + blends;
            Focus f;
            if (n == 0) return f;
            int k = Pick(m_Rng, n);
            bool all = Chance(m_Rng, 0.2f);
            if (k < (int)d.shapes.size()) {
                f.shape = d.shapes[k].id;
                if (!all) f.part = Pick(m_Rng, 4);          // position, size, rotation, profile
                return f;
            }
            k -= (int)d.shapes.size();
            if (k < (int)d.hinges.size()) {
                f.hinge = d.hinges[k].id;
                if (!all) {
                    int k2 = Pick(m_Rng, Timing() ? 4 : 3);   // pivot, axis, angle, timing
                    f.part = k2 < 3 ? 4 + k2 : 8;
                }
                return f;
            }
            k -= (int)d.hinges.size();
            for (const Group& g : d.groups)
                if (d.GroupShapes(g.id).size() >= 2 && k-- == 0) { f.group = g.id; break; }
            return f;
        }

        // One descent step on the parameters `ps` (one block, see RandomBlock), with an SPSA gradient (simultaneous
        // perturbation stochastic approximation): every parameter in the block is pushed by +1 or -1 unit at random,
        // all at once, and the score there against the score here gives the slope along that direction. The step
        // goes downhill along it, by `lr` units at a typical slope (slopes are measured against their running RMS)
        // and at most twice that (the paper's gradient clipping).
        //
        // This is where the paper and this code differ: there, the score is differentiable code and every step
        // follows its exact gradient over all parameters at once. A sampled distance field score has no gradient to
        // take, and a two-scoring SPSA estimate over every parameter at once is too noisy to descend on (it measured
        // worse than standing still). So each step estimates the gradient of one block (randomized block coordinate
        // descent), and a step is only taken if it lowers the score: whichever of the probe and the step is better
        // replaces the design when it beats it. The current score is known exactly, so one probe gives the slope (a
        // one-sided estimate): two scorings per step.
        bool Search::DescentStep(State& cur, const std::vector<Param>& ps, float lr) {
            if (ps.empty() || !cur.score.valid) return false;
            std::vector<float> delta(ps.size());
            for (float& x : delta) x = Chance(m_Rng, 0.5f) ? 1.0f : -1.0f;
            State probe;
            probe.design = cur.design;
            Step(probe.design, ps, delta);
            if (!Sanitize(probe.design)) return false;
            probe.score = Evaluate(probe.design, nullptr);
            if (!probe.score.valid) return false;
            Consider(probe.design, probe.score);       // a real design: it may be the best so far
            const float here = Objective(cur.score);
            float slope = Objective(probe.score) - here;
            m_SpsaRms = m_SpsaRms > 0.0f ? sqrtf(0.9f * m_SpsaRms * m_SpsaRms + 0.1f * slope * slope) : fabsf(slope);
            State* best = &probe;
            State next;
            if (m_SpsaRms > 1e-12f) {
                float g = std::clamp(slope / m_SpsaRms, -2.0f, 2.0f);
                std::vector<float> step(ps.size());
                for (size_t i = 0; i < ps.size(); i++) step[i] = -lr * g * delta[i];
                next.design = cur.design;
                Step(next.design, ps, step);
                if (Sanitize(next.design)) {
                    next.score = Evaluate(next.design, nullptr);
                    if (next.score.valid) {
                        Consider(next.design, next.score);
                        if (Objective(next.score) < Objective(probe.score)) best = &next;
                    }
                }
            }
            if (Objective(best->score) >= here) return false;
            cur.design = std::move(best->design);
            cur.score = best->score;
            return true;
        }

        // The paper's LocalOptimize, before a rewrite is judged: descent steps on just the parameters the rewrite
        // created or changed, so a good idea that landed slightly off isn't judged by its first guess. The paper
        // takes one exact gradient step; the estimated steps here are weaker, so up to `lookaheadSteps` are taken,
        // stopping at the first that doesn't help.
        void Search::Lookahead(State& cand, const Focus& f) {
            std::vector<Param> ps = Params(cand.design, &f);
            if (ps.empty()) return;
            for (int k = 0; k < m_P.lookaheadSteps && m_Evaluations + 2 <= m_P.iterations; k++)
                if (!DescentStep(cand, ps, m_P.stepSize)) break;
        }

        // ---- rewrites: the discrete half of SRD ----------------------------------------------------------------------

        // Draws one rewrite for design d (with the diagnostics of its score), all its random choices made. False
        // when the drawn kind doesn't apply to this design. Rewrite kinds are drawn with fixed weights (the paper
        // draws them about evenly; additions are made a little more likely here, since they carry most early progress).
        bool Search::Sample(const Design& d, const Diagnostics& dg, Rewrite* out) {
            float weight[(int)Rule::Count] = {
                2.0f, 1.0f, 1.5f, 0.6f, 0.8f,     // AddShape, AddCut, RemoveShape, ToggleOp, MoveShape
                1.0f, 1.0f, 0.5f, 0.8f,           // AddBody, SplitBody, RemoveBody, MergeBody
                0.5f, 0.4f, 0.3f,                 // AddFeature, DissolveFeature, ToggleFeature
                1.0f, 0.3f, 0.6f, 1.2f,           // AddHinge, RemoveHinge, Rehang, TurnHinge
                1.0f, 1.2f,                       // RemoveStray, CarveClash
                0.8f, 0.3f                        // Retime, OwnOpening
            };
            auto off = [&](Rule r) { weight[(int)r] = 0.0f; };
            if (!m_P.allowCuts) { off(Rule::AddCut); off(Rule::ToggleOp); off(Rule::ToggleFeature); off(Rule::CarveClash); }
            if (!m_P.allowFeatures) { off(Rule::AddFeature); off(Rule::ToggleFeature); }
            if (dg.excess.empty()) off(Rule::RemoveStray);
            if (dg.clashes.empty() || m_P.collisionWeight <= 0.0f) off(Rule::CarveClash);
            if (d.hinges.empty()) { off(Rule::RemoveHinge); off(Rule::Rehang); off(Rule::TurnHinge); off(Rule::MergeBody); }
            if (d.hinges.empty() || !Timing()) { off(Rule::Retime); off(Rule::OwnOpening); }
            float sum = 0.0f;
            for (float w : weight) sum += w;
            float pick = Uniform(m_Rng) * sum;
            int k = 0;
            while (k < (int)Rule::Count - 1 && pick >= weight[k]) pick -= weight[k++];

            Rewrite r;
            r.rule = (Rule)k;
            std::vector<int> bodies = d.Bodies();
            const int nBodies = (int)bodies.size();
            auto randomBody = [&]() { return d.groups[bodies[Pick(m_Rng, nBodies)]].id; };
            std::vector<int> features;
            for (const Group& g : d.groups)
                if (g.parent >= 0) features.push_back(g.id);
            switch (r.rule) {
                case Rule::AddShape: {
                    int g;
                    if (!FitRegion(d, dg, false, &r.sf, &g)) return false;
                    int body = d.groups[g].id;
                    r.group = body;
                    // sometimes into one of the body's blended features
                    std::vector<int> feats;
                    for (int id : features) {
                        const Group& gr = d.groups[d.GroupIndex(id)];
                        if (gr.op == Op::Union && d.BodyOf(id) == body) feats.push_back(id);
                    }
                    if (!feats.empty() && Chance(m_Rng, 0.3f)) r.group = feats[Pick(m_Rng, (int)feats.size())];
                    r.hasShape = true;
                    r.uses = { Key('G', body), Key('G', r.group) };
                    break;
                }
                case Rule::AddCut: {
                    int g;
                    if (!FitRegion(d, dg, true, &r.sf, &g)) return false;
                    r.group = d.groups[g].id;
                    r.hasShape = true;
                    r.uses = { Key('G', r.group) };
                    break;
                }
                case Rule::RemoveShape:
                case Rule::ToggleOp: {
                    if (d.shapes.empty()) return false;
                    int s = Pick(m_Rng, (int)d.shapes.size());
                    r.shape = d.shapes[s].id;
                    r.owns = { Key('S', r.shape) };
                    r.uses = { Key('G', d.ShapeBody(s)) };
                    break;
                }
                case Rule::MoveShape: {
                    if (nBodies < 2 || d.shapes.empty()) return false;
                    int s = Pick(m_Rng, (int)d.shapes.size());
                    int from = d.ShapeBody(s);
                    r.shape = d.shapes[s].id;
                    r.other = randomBody();
                    if (r.other == from) return false;
                    r.owns = { Key('S', r.shape) };
                    r.uses = { Key('G', from), Key('G', r.other) };
                    break;
                }
                case Rule::AddBody: {
                    // a new piece where material is missing, hinged to the nearest body
                    int g;
                    if (!FitRegion(d, dg, false, &r.sf, &g)) return false;
                    r.group = d.groups[g].id;
                    Design tmp = d;
                    int body = tmp.groups[tmp.AddGroup("Body", -1)].id;
                    tmp.AddShape(r.sf, body, "Shape");
                    r.pivot = PivotBetween(tmp, r.group, body, BodyHint(tmp, r.group), r.sf.position);
                    ChooseHinge(r.pivot, r.sf.position, &r.axis, &r.angle);
                    r.hasShape = r.hasHinge = true;
                    r.uses = { Key('G', r.group) };
                    break;
                }
                case Rule::SplitBody: {
                    // one shape of a body becomes a body of its own, hinged to what is left
                    if (nBodies < 1) return false;
                    int b = randomBody();
                    std::vector<int> mat;
                    for (int s : d.DirectShapes(b))
                        if (d.AddsMaterial(s)) mat.push_back(s);
                    if (mat.size() < 2) return false;
                    int s = mat[Pick(m_Rng, (int)mat.size())];
                    r.shape = d.shapes[s].id;
                    Design tmp = d;
                    int body = tmp.groups[tmp.AddGroup("Body", -1)].id;
                    tmp.MoveShape(s, body);
                    Vector3 at = d.shapes[s].sf.position;
                    r.pivot = PivotBetween(tmp, b, body, BodyHint(tmp, b), at);
                    ChooseHinge(r.pivot, at, &r.axis, &r.angle);
                    r.hasHinge = true;
                    r.owns = { Key('S', r.shape) };
                    r.uses = { Key('G', b) };
                    break;
                }
                case Rule::RemoveBody: {
                    if (nBodies < 2) return false;
                    r.group = randomBody();
                    r.owns = { Key('G', r.group), TreeKey };
                    break;
                }
                case Rule::MergeBody: {
                    const Bakugan::Hinge& h = d.hinges[Pick(m_Rng, (int)d.hinges.size())];
                    r.group = h.child;
                    r.owns = { Key('G', h.child), TreeKey };
                    r.uses = { Key('G', h.parent) };
                    break;
                }
                case Rule::AddFeature: {
                    // one or two shapes of a group become a blended feature
                    int g = d.groups[Pick(m_Rng, (int)d.groups.size())].id;
                    std::vector<int> direct = d.DirectShapes(g);
                    if (direct.size() < 2) return false;
                    std::shuffle(direct.begin(), direct.end(), m_Rng);
                    r.shapes = { d.shapes[direct[0]].id };
                    if (Chance(m_Rng, 0.6f)) r.shapes.push_back(d.shapes[direct[1]].id);
                    r.value = Uniform(m_Rng, 0.01f, 0.04f) * m_T.L;
                    for (int id : r.shapes) r.owns.push_back(Key('S', id));
                    r.uses = { Key('G', g) };
                    break;
                }
                case Rule::DissolveFeature:
                case Rule::ToggleFeature: {
                    if (features.empty()) return false;
                    r.group = features[Pick(m_Rng, (int)features.size())];
                    r.owns = { Key('G', r.group) };
                    r.uses = { Key('G', d.BodyOf(r.group)) };
                    break;
                }
                case Rule::AddHinge: {
                    // hang a loose body on the nearest body it can hang from
                    std::vector<int> roots = Roots(d);
                    if (roots.size() < 2) return false;
                    r.group = roots[Pick(m_Rng, (int)roots.size())];
                    Vector3 at = BodyHint(d, r.group);
                    int g = NearestBody(d, at, r.group);
                    if (g < 0) return false;
                    r.other = d.groups[g].id;
                    if (!d.CanHinge(r.other, r.group)) {
                        r.other = randomBody();
                        if (!d.CanHinge(r.other, r.group)) return false;
                    }
                    r.pivot = PivotBetween(d, r.other, r.group, BodyHint(d, r.other), at);
                    ChooseHinge(r.pivot, at, &r.axis, &r.angle);
                    r.hasHinge = true;
                    r.owns = { TreeKey, Key('G', r.group) };
                    r.uses = { Key('G', r.other) };
                    break;
                }
                case Rule::RemoveHinge: {
                    r.hinge = d.hinges[Pick(m_Rng, (int)d.hinges.size())].id;
                    r.owns = { TreeKey, Key('H', r.hinge) };
                    break;
                }
                case Rule::Rehang: {
                    // hang a body from another one
                    const Bakugan::Hinge& h = d.hinges[Pick(m_Rng, (int)d.hinges.size())];
                    r.hinge = h.id;
                    r.other = randomBody();
                    if (r.other == h.parent || !d.CanHinge(r.other, h.child, d.HingeIndex(h.id))) return false;
                    r.pivot = PivotBetween(d, r.other, h.child, BodyHint(d, r.other), BodyHint(d, h.child));
                    r.owns = { TreeKey, Key('H', r.hinge) };
                    r.uses = { Key('G', r.other) };
                    break;
                }
                case Rule::TurnHinge: {
                    // a large turn the descent can't make in small steps: anywhere, or a quarter / half turn on
                    const Bakugan::Hinge& h = d.hinges[Pick(m_Rng, (int)d.hinges.size())];
                    r.hinge = h.id;
                    r.angle = Chance(m_Rng, 0.5f) ? Uniform(m_Rng, -PI, PI)
                                                  : WrapAngle(h.end - h.start + (float)(1 + Pick(m_Rng, 3)) * PI * 0.5f);
                    r.owns = { Key('H', r.hinge) };
                    break;
                }
                case Rule::Retime: {
                    // a new window for when a hinge turns: anywhere, or just before or after another hinge's (folding
                    // one part out of the way of another first is a jump the descent can't make in small steps)
                    const Bakugan::Hinge& h = d.hinges[Pick(m_Rng, (int)d.hinges.size())];
                    r.hinge = h.id;
                    r.hasTiming = true;
                    r.ownOpening = h.ownOpening;
                    r.t0 = h.t0; r.t1 = h.t1; r.o0 = h.o0; r.o1 = h.o1;
                    bool opening = h.ownOpening && Chance(m_Rng, 0.5f);
                    float w = Uniform(m_Rng, 0.2f, 0.8f), a = Uniform(m_Rng, 0.0f, 1.0f - w);
                    if (d.hinges.size() >= 2 && Chance(m_Rng, 0.5f)) {
                        const Bakugan::Hinge& o = d.hinges[Pick(m_Rng, (int)d.hinges.size())];
                        float oa = opening ? o.o0 : o.t0, ob = opening ? o.o1 : o.t1;
                        a = Chance(m_Rng, 0.5f) ? ob : oa - w;                 // after it, or before it
                        a = std::clamp(a, 0.0f, 1.0f - w);
                    }
                    (opening ? r.o0 : r.t0) = a;
                    (opening ? r.o1 : r.t1) = a + w;
                    r.owns = { Key('H', r.hinge) };
                    break;
                }
                case Rule::OwnOpening: {
                    // a hinge opens on its own timing (a window of the opening motion), or goes back to retracing its
                    // closing in reverse
                    const Bakugan::Hinge& h = d.hinges[Pick(m_Rng, (int)d.hinges.size())];
                    r.hinge = h.id;
                    r.hasTiming = true;
                    r.t0 = h.t0; r.t1 = h.t1;
                    r.ownOpening = !h.ownOpening;
                    float w = Uniform(m_Rng, 0.2f, 0.8f), a = Uniform(m_Rng, 0.0f, 1.0f - w);
                    r.o0 = a;
                    r.o1 = a + w;
                    r.owns = { Key('H', r.hinge) };
                    break;
                }
                case Rule::RemoveStray: {
                    // the shape that put material off the mesh at a spot the score found: removed, or shrunk towards
                    // the mesh
                    const Spot& sp = dg.excess[Pick(m_Rng, (int)dg.excess.size())];
                    if (sp.body < 0 || sp.body >= (int)d.groups.size() || d.groups[sp.body].parent >= 0) return false;
                    int body = d.groups[sp.body].id;
                    Vector3 q = d.Transforms(0.0f, false)[sp.body].ApplyInverse(sp.p);
                    int best = -1;
                    float bd = FLT_MAX;
                    for (int s : d.GroupShapes(body)) {
                        if (!d.AddsMaterial(s)) continue;
                        float dd = d.shapes[s].sf.Distance(q);
                        if (dd < bd) { bd = dd; best = s; }
                    }
                    if (best < 0) return false;
                    r.shape = d.shapes[best].id;
                    if (Chance(m_Rng, 0.5f)) {
                        r.sf = d.shapes[best].sf;
                        Vector3 toward = Vector3Subtract(m_T.centre, r.sf.position);
                        r.sf.size = Vector3Scale(r.sf.size, Uniform(m_Rng, 0.5f, 0.85f));
                        r.sf.position = Vector3Add(r.sf.position, Vector3Scale(toward, Uniform(m_Rng, 0.0f, 0.3f)));
                        r.hasShape = true;
                    }
                    r.owns = { Key('S', r.shape) };
                    r.uses = { Key('G', body) };
                    break;
                }
                case Rule::CarveClash: {
                    // a cut in one of two overlapping bodies, fitted to where they overlap
                    int k2 = Pick(m_Rng, (int)dg.clashes.size());
                    int ga = dg.clashes[k2].body, gb = dg.clashOther[k2];
                    if (ga < 0 || gb < 0 || ga >= (int)d.groups.size() || gb >= (int)d.groups.size()) return false;
                    if (Chance(m_Rng, 0.5f)) std::swap(ga, gb);
                    std::vector<Rigid> xf = d.Transforms(dg.clashT[k2], true, dg.clashOpening[k2] ? Motion::Opening : Motion::Closing);
                    BodyField fa = BodyField::Body(d, d.groups[ga].id), fb = BodyField::Body(d, d.groups[gb].id);
                    if (fa.Empty() || fb.Empty()) return false;
                    float rho = m_T.L * Uniform(m_Rng, 0.04f, 0.15f);
                    std::vector<Vector3> keep;
                    for (int i = 0; i < 300; i++) {
                        Vector3 p;
                        do p = { Uniform(m_Rng, -1, 1), Uniform(m_Rng, -1, 1), Uniform(m_Rng, -1, 1) };
                        while (Vector3LengthSqr(p) > 1.0f);
                        p = Vector3Add(dg.clashes[k2].p, Vector3Scale(p, rho));
                        Vector3 qa = xf[ga].ApplyInverse(p);
                        if (fa.Distance(qa) < 0.0f && fb.Distance(xf[gb].ApplyInverse(p)) < 0.0f) keep.push_back(qa);
                    }
                    if (keep.size() < 10) return false;
                    r.sf = BoxAround(keep, 0.02f, 0.015f * m_T.L);
                    r.sf.size = Vector3Scale(r.sf.size, Uniform(m_Rng, 1.05f, 1.3f));
                    r.sf.rounding = Uniform(m_Rng);
                    r.hasShape = true;
                    r.group = d.groups[ga].id;
                    r.uses = { Key('G', r.group) };
                    break;
                }
                case Rule::Count:
                    return false;
            }
            *out = std::move(r);
            return true;
        }

        // Applies a rewrite (sampled for this design, or for one it was combined into). False when it no longer
        // applies: what it acts on is gone, or it would leave a body without material. The shapes and hinges it
        // creates or changes go into *f, for the lookahead.
        // Applies a rewrite, then the timing its lookahead found (or it drew) to the hinge it touched.
        bool Search::Apply(Design& d, const Rewrite& r, Focus* f) const {
            if (!ApplyRule(d, r, f)) return false;
            if (r.hasTiming) {
                int h = d.HingeIndex(f->hinge >= 0 ? f->hinge : r.hinge);
                if (h >= 0) {
                    Bakugan::Hinge& hg = d.hinges[h];
                    hg.t0 = r.t0; hg.t1 = r.t1;
                    hg.ownOpening = r.ownOpening;
                    hg.o0 = r.o0; hg.o1 = r.o1;
                    ClampTiming(hg);
                    if (f->hinge < 0) f->hinge = hg.id;
                }
            }
            return true;
        }

        bool Search::ApplyRule(Design& d, const Rewrite& r, Focus* f) const {
            auto isBody = [&](int id) {
                int g = d.GroupIndex(id);
                return g >= 0 && d.groups[g].parent < 0;
            };
            auto roomShape = [&]() { return (int)d.shapes.size() < m_P.maxShapes; };
            auto roomBody = [&]() { return (int)d.Bodies().size() < m_P.maxBodies; };
            // the shape's values found by the lookahead
            auto tuned = [&](int s) {
                if (r.hasShape) d.shapes[s].sf = r.sf;
                f->shape = d.shapes[s].id;
            };
            auto lastMaterial = [&](int s) { return d.AddsMaterial(s) && MaterialShapes(d, d.ShapeBody(s)) <= 1; };
            switch (r.rule) {
                case Rule::AddShape: {
                    if (d.GroupIndex(r.group) < 0 || !roomShape()) return false;
                    f->shape = d.shapes[d.AddShape(r.sf, r.group, "Shape")].id;
                    return true;
                }
                case Rule::AddCut:
                case Rule::CarveClash: {
                    if (!isBody(r.group) || !roomShape()) return false;
                    f->shape = d.shapes[d.AddShape(r.sf, r.group, "Cut", Op::Cut)].id;
                    return true;
                }
                case Rule::RemoveShape: {
                    int s = d.ShapeIndex(r.shape);
                    if (s < 0 || d.shapes.size() <= 1 || lastMaterial(s)) return false;
                    d.RemoveShape(s);
                    return true;
                }
                case Rule::ToggleOp: {
                    int s = d.ShapeIndex(r.shape);
                    if (s < 0 || lastMaterial(s)) return false;
                    Shape& sh = d.shapes[s];
                    sh.op = sh.op == Op::Union ? Op::Cut : Op::Union;
                    sh.color = sh.op == Op::Cut ? Color{ 225, 90, 90, 255 } : d.groups[d.GroupIndex(sh.group)].color;
                    tuned(s);
                    return true;
                }
                case Rule::MoveShape: {
                    int s = d.ShapeIndex(r.shape);
                    if (s < 0 || !isBody(r.other) || d.ShapeBody(s) == r.other || lastMaterial(s)) return false;
                    d.MoveShape(s, r.other);
                    if (d.shapes[s].op == Op::Union) d.shapes[s].color = d.groups[d.GroupIndex(r.other)].color;
                    tuned(s);
                    return true;
                }
                case Rule::AddBody: {
                    if (!isBody(r.group) || !roomShape() || !roomBody()) return false;
                    int body = d.groups[d.AddGroup("Body", -1)].id;
                    f->shape = d.shapes[d.AddShape(r.sf, body, "Shape")].id;
                    if (!d.CanHinge(r.group, body)) return false;
                    int h = d.AddHinge(r.group, body);
                    SetupHinge(d, h, r.pivot, r.axis, r.angle, true);
                    f->hinge = d.hinges[h].id;
                    return true;
                }
                case Rule::SplitBody: {
                    int s = d.ShapeIndex(r.shape);
                    if (s < 0 || !roomBody()) return false;
                    int from = d.ShapeBody(s);
                    // a body's own material shape, and not its last
                    if (d.shapes[s].group != from || !d.AddsMaterial(s) || MaterialShapes(d, from) <= 1) return false;
                    int nb = d.AddGroup("Body", -1);
                    int body = d.groups[nb].id;
                    d.MoveShape(s, body);
                    d.shapes[s].color = d.groups[nb].color;
                    if (!d.CanHinge(from, body)) return false;
                    int h = d.AddHinge(from, body);
                    SetupHinge(d, h, r.pivot, r.axis, r.angle, true);
                    f->hinge = d.hinges[h].id;
                    return true;
                }
                case Rule::RemoveBody: {
                    if (!isBody(r.group) || d.Bodies().size() <= 1) return false;
                    RemoveBody(d, r.group);
                    return true;
                }
                case Rule::MergeBody: {
                    if (!isBody(r.group)) return false;
                    return MergeBody(d, r.group);
                }
                case Rule::AddFeature: {
                    int parent = -1;
                    for (int id : r.shapes) {
                        int s = d.ShapeIndex(id);
                        if (s < 0 || (parent >= 0 && d.shapes[s].group != parent)) return false;
                        parent = d.shapes[s].group;
                    }
                    if (parent < 0) return false;
                    int g = d.GroupIndex(d.GroupItems(r.shapes, {}));
                    if (g < 0 || d.groups[g].parent < 0) return false;
                    d.groups[g].blend = r.value;
                    return true;
                }
                case Rule::DissolveFeature: {
                    int g = d.GroupIndex(r.group);
                    if (g < 0 || d.groups[g].parent < 0) return false;
                    d.Ungroup(g);
                    return true;
                }
                case Rule::ToggleFeature: {
                    int g = d.GroupIndex(r.group);
                    if (g < 0 || d.groups[g].parent < 0) return false;
                    d.groups[g].op = d.groups[g].op == Op::Union ? Op::Cut : Op::Union;
                    return MaterialShapes(d, d.BodyOf(r.group)) > 0;
                }
                case Rule::AddHinge: {
                    if (!isBody(r.group) || !isBody(r.other) || d.ParentHinge(r.group) >= 0 || !d.CanHinge(r.other, r.group)) return false;
                    int h = d.AddHinge(r.other, r.group);
                    SetupHinge(d, h, r.pivot, r.axis, r.angle, true);
                    f->hinge = d.hinges[h].id;
                    return true;
                }
                case Rule::RemoveHinge: {
                    int h = d.HingeIndex(r.hinge);
                    if (h < 0) return false;
                    d.RemoveHinge(h);
                    return true;
                }
                case Rule::Rehang: {
                    int h = d.HingeIndex(r.hinge);
                    if (h < 0 || !isBody(r.other) || !d.CanHinge(r.other, d.hinges[h].child, h)) return false;
                    d.hinges[h].parent = r.other;
                    d.hinges[h].pivot = r.pivot;
                    if (r.hasHinge) SetupHinge(d, h, r.pivot, r.axis, r.angle, false);
                    f->hinge = r.hinge;
                    return true;
                }
                case Rule::TurnHinge: {
                    int h = d.HingeIndex(r.hinge);
                    if (h < 0) return false;
                    if (r.hasHinge) SetupHinge(d, h, r.pivot, r.axis, r.angle, false);
                    else d.hinges[h].end = d.hinges[h].start + r.angle;
                    f->hinge = r.hinge;
                    return true;
                }
                case Rule::Retime:
                case Rule::OwnOpening: {
                    if (d.HingeIndex(r.hinge) < 0) return false;
                    f->hinge = r.hinge;                  // Apply sets the timing
                    return true;
                }
                case Rule::RemoveStray: {
                    int s = d.ShapeIndex(r.shape);
                    if (s < 0) return false;
                    if (r.hasShape) {
                        tuned(s);
                        return true;
                    }
                    if (lastMaterial(s) || !d.AddsMaterial(s)) return false;
                    d.RemoveShape(s);
                    return true;
                }
                case Rule::Count:
                    break;
            }
            return false;
        }

        // Writes what the lookahead settled on back into the rewrite, so applying it again (combined with other
        // rewrites) gives the tuned shape and hinge rather than the first guess.
        void Search::Remember(Rewrite& r, const Design& d, const Focus& f) const {
            int s = d.ShapeIndex(f.shape);
            if (s >= 0) {
                r.sf = d.shapes[s].sf;
                r.hasShape = true;
            }
            int h = d.HingeIndex(f.hinge);
            if (h >= 0) {
                const Bakugan::Hinge& hg = d.hinges[h];
                r.pivot = hg.pivot;
                r.axis = hg.axis;
                r.angle = hg.end - hg.start;
                r.hasHinge = true;
                r.hasTiming = true;
                r.t0 = hg.t0; r.t1 = hg.t1;
                r.ownOpening = hg.ownOpening;
                r.o0 = hg.o0; r.o1 = hg.o1;
            }
        }

        // One rewrite round: score the current design exactly (its diagnostics guide the sampling), draw rewrites,
        // give each a lookahead, and apply every one that improved the objective by more than a small threshold, best
        // first, skipping any that conflict with one already taken: all at once (the paper's greedy selection). Should
        // the combination score worse than the best rewrite alone (rewrites can interact), that one alone is taken.
        // Returns how many rewrites were applied.
        int Search::RewriteRound(State& cur) {
            const int N = m_P.iterations;
            cur.score = Evaluate(cur.design, &cur.diag);
            if (!cur.score.valid) return 0;
            Consider(cur.design, cur.score);
            m_Score = cur.score;
            const float base = Objective(cur.score);
            const float threshold = fmaxf(1e-4f, 1e-3f * base);
            struct Kept {
                Rewrite r;
                State s;
            };
            std::vector<Kept> kept;
            int drawn = 0;
            for (int tries = 0; drawn < m_P.rewritesPerRound && tries < 4 * m_P.rewritesPerRound; tries++) {
                if (m_Evaluations >= N || m_C.Cancelled()) break;
                Rewrite r;
                if (!Sample(cur.design, cur.diag, &r)) continue;
                State cand;
                cand.design = cur.design;
                Focus f;
                if (!Apply(cand.design, r, &f) || !Sanitize(cand.design)) continue;
                drawn++;
                cand.score = Evaluate(cand.design, nullptr);
                if (!cand.score.valid) continue;
                Consider(cand.design, cand.score);
                // a lookahead step costs two scorings: spent on rewrites that landed close enough for it to matter
                bool close = base - Objective(cand.score) > -0.1f * base;
                if (close && (f.shape >= 0 || f.hinge >= 0)) Lookahead(cand, f);
                Remember(r, cand.design, f);
                r.gain = base - Objective(cand.score);
                if (r.gain > threshold) kept.push_back({ std::move(r), std::move(cand) });
            }
            if (kept.empty()) return 0;
            std::sort(kept.begin(), kept.end(), [](const Kept& a, const Kept& b) { return a.r.gain > b.r.gain; });
            std::vector<const Rewrite*> chosen;
            for (const Kept& k : kept) {
                bool clash = false;
                for (const Rewrite* c : chosen) clash = clash || Conflict(k.r, *c);
                if (!clash) chosen.push_back(&k.r);
            }
            if (chosen.size() > 1) {
                Design combined = cur.design;
                int applied = 0;
                for (const Rewrite* r : chosen) {
                    Design t = combined;
                    Focus f;
                    if (Apply(t, *r, &f) && Sanitize(t)) {
                        combined = std::move(t);
                        applied++;
                    }
                }
                if (applied > 1) {
                    AutoScore cs = Evaluate(combined, nullptr);
                    if (cs.valid) {
                        Consider(combined, cs);
                        if (Objective(cs) <= Objective(kept[0].s.score)) {
                            cur.design = std::move(combined);
                            cur.score = cs;
                            return applied;
                        }
                    }
                }
            }
            cur = std::move(kept[0].s);
            return 1;
        }

        // Keeps a design inside the space the search works in. False rejects it.
        bool Search::Sanitize(Design& d) const {
            d.Repair();
            // empty features go, and bodies that add no material (keeping the hinge tree whole)
            for (bool again = true; again;) {
                again = false;
                for (int g = 0; g < (int)d.groups.size(); g++) {
                    const Group& gr = d.groups[g];
                    if (gr.parent >= 0 && d.DirectShapes(gr.id).empty() && d.ChildGroups(gr.id).empty()) {
                        d.RemoveGroup(g);
                        again = true;
                        break;
                    }
                }
                if (again) continue;
                for (int g : d.Bodies()) {
                    if (MaterialShapes(d, d.groups[g].id) > 0) continue;
                    RemoveBody(d, d.groups[g].id);
                    again = true;
                    break;
                }
            }
            if (d.Bodies().empty() || d.shapes.empty()) return false;
            int bodies = (int)d.Bodies().size();
            if ((int)d.shapes.size() > m_P.maxShapes || bodies > m_P.maxBodies || bodies < m_P.minBodies) return false;
            for (const Group& g : d.groups)
                if (g.blend > m_MaxBlend) return false;      // bigger blends melt everything into one blob
            for (const Shape& s : d.shapes) {
                const SuperFrustum& f = s.sf;
                if (fminf(f.size.x, fminf(f.size.y, f.size.z)) < m_MinSize) return false;
                if (fmaxf(f.size.x, fmaxf(f.size.y, f.size.z)) > m_MaxSize) return false;
                if (f.dilation > m_MaxDilation) return false;
                if (Vector3Distance(f.position, m_T.centre) > m_Reach) return false;
            }
            for (Bakugan::Hinge& h : d.hinges) {
                h.hardware = false;
                if (Vector3Distance(h.pivot, m_T.centre) > m_Reach) return false;
            }
            return true;
        }

        void Search::Finalize(Design& d) const {
            for (Bakugan::Hinge& h : d.hinges) {
                auto it = m_Hardware.find(h.id);
                h.hardware = m_P.hingeHardware && (it == m_Hardware.end() || it->second);
            }
            d.Repair();
        }

        // The paper's projection to constraints: after a descent step, values are pulled back into their ranges (a
        // step should be clamped, not thrown away the way Sanitize would throw away the design).
        void Search::Project(Design& d) const {
            auto pull = [&](Vector3& p) {
                Vector3 off = Vector3Subtract(p, m_T.centre);
                float l = Vector3Length(off);
                if (l > m_Reach) p = Vector3Add(m_T.centre, Vector3Scale(off, 0.999f * m_Reach / l));
            };
            for (Shape& sh : d.shapes) {
                SuperFrustum& f = sh.sf;
                for (int a = 0; a < 3; a++) (&f.size.x)[a] = std::clamp((&f.size.x)[a], m_MinSize, m_MaxSize);
                f.rounding = std::clamp(f.rounding, 0.0f, 1.0f);
                f.taper = std::clamp(f.taper, 0.0f, 1.0f);
                f.bulge = std::clamp(f.bulge, -1.0f, 1.0f);
                f.dilation = std::clamp(f.dilation, 0.0f, m_MaxDilation);
                pull(f.position);
            }
            for (Group& g : d.groups) g.blend = std::clamp(g.blend, 0.0f, m_MaxBlend);
            for (Bakugan::Hinge& h : d.hinges) {
                pull(h.pivot);
                h.axis = Vector3Length(h.axis) > 1e-6f ? Vector3Normalize(h.axis) : Vector3{ 0, 1, 0 };
                if (Timing()) ClampTiming(h);
            }
        }

        // Every design scored exactly is a candidate for the result: the start, every probe, every rewrite.
        void Search::Consider(const Design& d, const AutoScore& s) {
            if (!s.valid) return;
            if (m_Best.score.valid && Objective(s) >= Objective(m_Best.score) - 1e-6f) return;
            m_Best.design = d;
            m_Best.score = s;
            if (s.silhouettes) {
                // Consider always follows the scoring of this design, so the last outlines are its
                m_BestOpen = m_LastOpen;
                m_BestClosed = m_LastClosed;
            }
            Publish(false);
        }

        void Search::Publish(bool force) {
            if (!m_Live) return;
            Clock::time_point now = Clock::now();
            if (!force && now - m_LastPublish < std::chrono::milliseconds(250)) return;
            m_LastPublish = now;
            Design d = m_Best.design;
            Finalize(d);
            std::vector<unsigned char> views;
            int vw = 0, vh = 0;
            if (m_Best.score.silhouettes) ViewsImage(&views, &vw, &vh);
            std::lock_guard<std::mutex> l(m_Live->mutex);
            m_Live->views = std::move(views);
            m_Live->viewsWidth = vw;
            m_Live->viewsHeight = vh;
            m_Live->best = std::move(d);
            m_Live->score = m_Best.score;
            m_Live->iteration = m_Evaluations;
            m_Live->version++;
        }

        // ---- starting points -------------------------------------------------------------------------------------------

        // Splits the inside of the mesh into k pieces (k-means), gives each a box-ish superfrustum along its principal
        // axes, and hinges them in a tree (nearest neighbours first, from the biggest piece) with each piece folding
        // towards the middle of the mesh.
        Design Search::Seed() {
            Design d;
            int k = std::clamp(std::max(m_P.seedBodies, m_P.minBodies), 1, std::max(1, m_P.maxBodies));
            std::vector<Vector3> pts = m_T.inside;
            std::shuffle(pts.begin(), pts.end(), m_Rng);
            if (pts.size() > 2500) pts.resize(2500);
            k = std::min(k, (int)pts.size() / 20);
            k = std::max(k, 1);
            // k-means++
            std::vector<Vector3> centres = { pts[Pick(m_Rng, (int)pts.size())] };
            std::vector<float> dist(pts.size(), FLT_MAX);
            while ((int)centres.size() < k) {
                double total = 0.0;
                for (size_t i = 0; i < pts.size(); i++) {
                    dist[i] = fminf(dist[i], Vector3DistanceSqr(pts[i], centres.back()));
                    total += dist[i];
                }
                double r = Uniform(m_Rng) * total;
                size_t pick = 0;
                for (; pick + 1 < pts.size() && r > dist[pick]; pick++) r -= dist[pick];
                centres.push_back(pts[pick]);
            }
            std::vector<int> label(pts.size(), 0);
            for (int it = 0; it < 15; it++) {
                for (size_t i = 0; i < pts.size(); i++) {
                    float best = FLT_MAX;
                    for (int c = 0; c < k; c++) {
                        float dd = Vector3DistanceSqr(pts[i], centres[c]);
                        if (dd < best) { best = dd; label[i] = c; }
                    }
                }
                std::vector<Vector3> sum(k, { 0, 0, 0 });
                std::vector<int> count(k, 0);
                for (size_t i = 0; i < pts.size(); i++) { sum[label[i]] = Vector3Add(sum[label[i]], pts[i]); count[label[i]]++; }
                for (int c = 0; c < k; c++)
                    if (count[c] > 0) centres[c] = Vector3Scale(sum[c], 1.0f / count[c]);
            }
            std::vector<std::vector<Vector3>> cluster(k);
            for (size_t i = 0; i < pts.size(); i++) cluster[label[i]].push_back(pts[i]);
            std::vector<int> order;
            for (int c = 0; c < k; c++)
                if (cluster[c].size() >= 8) order.push_back(c);
            std::sort(order.begin(), order.end(), [&](int a, int b) { return cluster[a].size() > cluster[b].size(); });

            // one oriented, trimmed box per piece
            std::vector<int> bodyOf(k, -1);
            for (int c : order) {
                SuperFrustum f = BoxAround(cluster[c], 0.03f, 0.02f * m_T.L);
                f.rounding = 0.6f;
                int b = d.groups[d.AddGroup("Body", -1)].id;
                d.AddShape(f, b, "Shape");
                bodyOf[c] = b;
            }
            // hinge tree: Prim's algorithm over the gaps between pieces, from the biggest
            auto gapBetween = [&](int a, int b, Vector3* mid) {
                float best = FLT_MAX;
                size_t na = std::min(cluster[a].size(), (size_t)160), nb = std::min(cluster[b].size(), (size_t)160);
                for (size_t i = 0; i < na; i++)
                    for (size_t j = 0; j < nb; j++) {
                        float dd = Vector3DistanceSqr(cluster[a][i], cluster[b][j]);
                        if (dd < best) { best = dd; *mid = Vector3Lerp(cluster[a][i], cluster[b][j], 0.5f); }
                    }
                return best;
            };
            std::vector<int> tree;
            if (!order.empty()) tree.push_back(order[0]);
            while (tree.size() < order.size()) {
                float best = FLT_MAX;
                int bp = -1, bc = -1;
                Vector3 bmid = { 0, 0, 0 };
                for (int a : tree)
                    for (int c : order) {
                        if (std::find(tree.begin(), tree.end(), c) != tree.end()) continue;
                        Vector3 mid = { 0, 0, 0 };
                        float g = gapBetween(a, c, &mid);
                        if (g < best) { best = g; bp = a; bc = c; bmid = mid; }
                    }
                if (bc < 0) break;
                tree.push_back(bc);
                int h = AttachHinge(d, bodyOf[bp], bodyOf[bc], bmid, centres[bc]);
                int hi = d.HingeIndex(h);
                if (hi >= 0) {
                    // seeds always fold towards the middle
                    Bakugan::Hinge& hg = d.hinges[hi];
                    Vector3 out = Vector3Normalize(Vector3Subtract(centres[bc], bmid));
                    Vector3 in = Vector3Subtract(m_T.centre, bmid);
                    Vector3 axis = Vector3CrossProduct(out, in);
                    if (Vector3Length(axis) > 1e-3f * fmaxf(Vector3Length(in), 1e-3f)) {
                        hg.axis = Vector3Normalize(axis);
                        hg.end = 0.85f * acosf(std::clamp(Vector3DotProduct(out, Vector3Normalize(in)), -1.0f, 1.0f));
                    }
                }
            }
            return d;
        }

        Design Search::FromCurrent(const Design& start) {
            Design d = start;
            d.Repair();
            for (const Bakugan::Hinge& h : d.hinges) m_Hardware[h.id] = h.hardware;
            m_P.maxShapes = std::max(m_P.maxShapes, (int)d.shapes.size());
            m_P.maxBodies = std::max(m_P.maxBodies, (int)d.Bodies().size());
            m_P.minBodies = std::min(m_P.minBodies, (int)d.Bodies().size());
            // whatever the start already does is allowed (with a little room), or no change to it could be kept
            for (const Shape& s : d.shapes) {
                const SuperFrustum& f = s.sf;
                m_Reach = fmaxf(m_Reach, Vector3Distance(f.position, m_T.centre) + 0.1f * m_T.L);
                m_MinSize = fminf(m_MinSize, 0.9f * fminf(f.size.x, fminf(f.size.y, f.size.z)));
                m_MaxSize = fmaxf(m_MaxSize, 1.1f * fmaxf(f.size.x, fmaxf(f.size.y, f.size.z)));
                m_MaxDilation = fmaxf(m_MaxDilation, 1.1f * f.dilation);
            }
            for (const Group& g : d.groups) m_MaxBlend = fmaxf(m_MaxBlend, 1.1f * g.blend);
            for (const Bakugan::Hinge& h : d.hinges) m_Reach = fmaxf(m_Reach, Vector3Distance(h.pivot, m_T.centre) + 0.1f * m_T.L);
            return d;
        }

        // ---- the run ---------------------------------------------------------------------------------------------------

        // Stochastic Rewrite Descent: rounds of descent steps (structure fixed) and rewrite batches (structure
        // changing), until the scorings run out. The descent step size follows the success rate of the steps and goes
        // back to full size after every rewrite round, when the structure has changed under it (as the paper's code
        // raises its step after each round). Every design scored exactly is a candidate for the result; should the
        // search wander far from the best, it starts again from there.
        AutoResult Search::Run(const Design& start) {
            AutoResult res;
            if (!Prepare()) {
                res.message = m_C.Cancelled() ? "Stopped." : "The mesh has no usable inside (is it closed?).";
                res.cancelled = m_C.Cancelled();
                return res;
            }
            MakeSamples();
            bool current = m_P.fromCurrent && !start.shapes.empty();
            State cur;
            cur.design = current ? FromCurrent(start) : Seed();
            // a seed with fewer pieces than asked for (a small mesh) sets the floor
            m_P.minBodies = std::min(m_P.minBodies, (int)cur.design.Bodies().size());
            if (!Sanitize(cur.design))
                for (Bakugan::Hinge& h : cur.design.hinges) h.hardware = false;
            Project(cur.design);
            m_Ramp = 0.1f;
            cur.score = Evaluate(cur.design, &cur.diag);
            if (!cur.score.valid) { res.message = "Could not score the starting design."; return res; }
            m_Best = cur;
            m_BestOpen = m_LastOpen;
            m_BestClosed = m_LastClosed;
            Publish(true);

            const int N = std::max(0, m_P.iterations);
            m_P.iterations = N;
            // evenly spaced snapshots of the best design, for a recording of the run
            int recordCount = 0, recorded = 0;
            if (m_Live) {
                std::lock_guard<std::mutex> l(m_Live->mutex);
                recordCount = m_Live->recordFrames;
                m_Live->frames.clear();
                m_Live->iterations = N;
                m_Live->history.clear();
                m_Live->history.push_back(m_Best.score.total);
            }
            auto record = [&](bool last) {
                if (recordCount <= 0) return;
                while (recorded < recordCount &&
                       (last || (long long)m_Evaluations * std::max(1, recordCount - 1) >= (long long)recorded * std::max(1, N))) {
                    AutoFrame f;
                    f.design = m_Best.design;
                    Finalize(f.design);
                    f.score = m_Best.score;
                    f.iteration = std::min(m_Evaluations, N);
                    std::lock_guard<std::mutex> l(m_Live->mutex);
                    m_Live->frames.push_back(std::move(f));
                    recorded++;
                }
            };
            int historyEvery = std::max(1, N / 240), nextHistory = historyEvery;
            auto report = [&]() {
                m_C.SetProgress(N > 0 ? std::min(1.0f, (float)m_Evaluations / N) : 1.0f);
                record(false);
                if (m_Live && m_Evaluations >= nextHistory) {
                    nextHistory = m_Evaluations + historyEvery;
                    std::lock_guard<std::mutex> l(m_Live->mutex);
                    m_Live->history.push_back(m_Best.score.total);
                    m_Live->iteration = m_Evaluations;
                    m_Live->accepted = m_Accepted;
                    m_Live->rounds = m_Rounds;
                    m_Live->stepSize = m_Lr;
                }
            };
            record(false);

            m_Lr = m_P.stepSize;
            const float minLr = 0.05f * m_P.stepSize;
            m_C.SetStage("Searching");
            while (m_Evaluations < N && !m_C.Cancelled()) {
                // collisions start cheap and reach their full weight at 60% of the run: hinges and folds get to form
                // before overlaps between the bodies have to be carved or moved away
                m_Ramp = std::min(1.0f, 0.1f + 0.9f * ((float)m_Evaluations / N) / 0.6f);

                // descent, the structure fixed: steps on one random element's parameters at a time. A step that
                // doesn't help mostly means that element is already where it should be, not that the step is too
                // big, so the step size follows the success rate instead of runs of failures: it settles where about
                // one step in four helps (x1.25 on success, x0.93 otherwise)
                for (int k = 0; k < m_P.stepsPerRound && m_Evaluations + 2 <= N && !m_C.Cancelled(); k++) {
                    Focus block = RandomBlock(cur.design);
                    std::vector<Param> ps = Params(cur.design, &block);
                    bool helped = DescentStep(cur, ps, m_Lr);
                    m_Lr = std::clamp(m_Lr * (helped ? 1.25f : 0.93f), minLr, 2.0f * m_P.stepSize);
                    report();
                }
                if (m_Evaluations >= N || m_C.Cancelled()) break;

                // rewrites
                int applied = RewriteRound(cur);
                m_Accepted += applied;
                m_Rounds++;
                m_Lr = fmaxf(m_Lr, m_P.stepSize);          // a new structure: full size steps again
                // wandered off: go back to the best
                if (cur.score.valid && Objective(cur.score) > Objective(m_Best.score) + 0.02f) cur = m_Best;
                report();
            }
            // the descent since the last round has not been scored exactly
            if (!m_C.Cancelled() && N > 0) {
                AutoScore last = Evaluate(cur.design, nullptr);
                Consider(cur.design, last);
            }
            m_C.SetProgress(1.0f);
            if (!m_C.Cancelled()) record(true);
            Publish(true);
            if (m_Live) {
                std::lock_guard<std::mutex> l(m_Live->mutex);
                m_Live->history.push_back(m_Best.score.total);
                m_Live->iteration = m_Evaluations;
                m_Live->accepted = m_Accepted;
                m_Live->rounds = m_Rounds;
                m_Live->stepSize = m_Lr;
            }
            res.ok = true;
            res.cancelled = m_C.Cancelled();
            res.design = m_Best.design;
            Finalize(res.design);
            res.score = m_Best.score;
            const AutoScore& s = m_Best.score;
            const char* what = res.cancelled ? "Stopped, kept the best so far:" : (N == 0 ? "Scored:" : "Generated");
            std::string shape = Utils::Format("%d bodies, %d shapes, %d hinges", (int)res.design.Bodies().size(), (int)res.design.shapes.size(),
                                              (int)res.design.hinges.size());
            std::string fits = s.silhouettes
                ? Utils::Format("outlines cover %.0f%% of the mesh's (+%.0f%% extra) and %.0f%% of the sphere's (+%.0f%% extra), worst view %.0f%%",
                                s.viewCover * 100.0f, s.viewExtra * 100.0f, s.sphereViewCover * 100.0f, s.sphereViewExtra * 100.0f, s.worstView * 100.0f)
                : Utils::Format("covers %.0f%% of the mesh, %.0f%% outside it, sphere %.0f%%", s.fitCover * 100.0f, s.fitOutside * 100.0f,
                                s.sphereIoU * 100.0f);
            res.message = Utils::Format("%s %s: %s, overlapping %.1f%% open / %.1f%% closed%s (score %.3f; %d scorings, %d rewrites in %d rounds).",
                                        what, shape.c_str(), fits.c_str(), s.clashOpen * 100.0f, s.clashClosed * 100.0f,
                                        m_P.motionChecks > 0 ? Utils::Format(" / %.1f%% on the way", s.clashMotion * 100.0f).c_str() : "", s.total,
                                        m_Evaluations, m_Accepted, m_Rounds);
            return res;
        }
    }

    AutoResult AutoDesign(const TriMesh& mesh, const Design& start, const AutoParams& p, JobControl& c, std::shared_ptr<AutoLive> live) {
        Search s(mesh, p, c, std::move(live));
        return s.Run(start);
    }
}
