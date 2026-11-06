#include "autodesign.hpp"
#include "data/field.hpp"
#include "util/utils.hpp"
#include <raymath.h>
#include <algorithm>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
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

        // Turns q by a small random rotation (radians, standard deviation).
        Quaternion Jiggle(Quaternion q, float sigma, Rng& r) {
            Quaternion d = QuaternionFromAxisAngle(RandomUnit(r), Normal(r, sigma));
            return QuaternionNormalize(QuaternionMultiply(d, q));
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
            std::vector<Spot> clashes;   // inside two bodies: open (body = the first) ...
            std::vector<int> clashOther; // ... the second body (group index)
            std::vector<char> clashClosed;   // ... and whether it is in the closed pose
        };

        // What a structural change touched, so it can be tuned a little before it is judged.
        struct Focus {
            int shape = -1;              // shape id
            int hinge = -1;              // hinge id
        };

        struct State {
            Design design;
            AutoScore score;
            Diagnostics diag;
        };

        // ---- the search ------------------------------------------------------------------------------------------------

        class Search {
        public:
            Search(const TriMesh& mesh, const AutoParams& p, JobControl& c, std::shared_ptr<AutoLive> live)
                : m_Mesh(mesh), m_P(p), m_C(c), m_Live(std::move(live)), m_Rng(p.seed),
                  m_Pool(std::max(0, (int)std::thread::hardware_concurrency() - 1)) {
                m_P.maxBodies = std::max(1, m_P.maxBodies);
                m_P.minBodies = std::clamp(m_P.minBodies, 1, m_P.maxBodies);
            }

            AutoResult Run(const Design& start);
        private:
            bool Prepare();
            void MakeSamples();
            Design Seed();
            Design FromCurrent(const Design& start);
            AutoScore Evaluate(const Design& d, Diagnostics* diag);
            bool Propose(Design& d, const Diagnostics& dg, float progress, Focus* focus, bool* small);
            bool Perturb(Design& d, const Focus& f, float scale);
            bool PerturbShape(Design& d, int s, float scale);
            bool PerturbHinge(Design& d, int h, float scale);
            bool Structural(Design& d, const Diagnostics& dg, Focus* focus);
            bool Sanitize(Design& d) const;
            void Finalize(Design& d) const;
            void Publish(const State& best, int iteration, bool force);

            // grammar helpers
            SuperFrustum NewShape(Vector3 at, bool cut);
            bool PlaceSpot(const Design& d, const Diagnostics& dg, bool forCut, Vector3* at, int* bodyGroup, bool* sphere = nullptr);
            bool FitRegion(const Design& d, const Diagnostics& dg, bool cut, SuperFrustum* f, int* bodyGroup);
            int NearestBody(const Design& d, Vector3 p, int except = -1) const;   // group index
            Vector3 BodyHint(const Design& d, int bodyId) const;                   // a point inside the body
            Vector3 PivotBetween(const Design& d, int bodyA, int bodyB, Vector3 a, Vector3 b) const;
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
            float m_Step = 1.0f;             // size of small changes: grows while they keep helping, shrinks while they don't
            std::unique_ptr<InsideTest> m_Inside;
            MeshField m_MeshField;
            // the box the search stays in (shared by every design, so a start design outside it is never stuck)
            float m_Reach = 0.0f;            // how far shapes and pivots may be from the mesh's centre
            float m_MinSize = 0.0f, m_MaxSize = 0.0f, m_MaxDilation = 0.0f, m_MaxBlend = 0.0f;
            AutoScore m_Score;               // of the design moves are proposed for (its sphere)
            std::unordered_map<int, bool> m_Hardware;   // hinge id -> hardware flag of a design we started from
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
            int q = std::clamp(m_P.quality, 0, 2);
            Rng r(m_P.seed * 7919u + 17u);
            for (int i = 0; i < closedCount[q]; i++) m_S.closedUnit.push_back({ Uniform(r), Uniform(r), Uniform(r) });
            for (int i = 0; i < closedCount[q]; i++) m_S.openUnit.push_back({ Uniform(r), Uniform(r), Uniform(r) });
            for (int i = 0; i < sphereCount[q]; i++) {
                Vector3 u = { Uniform(r, -1.35f, 1.35f), Uniform(r, -1.35f, 1.35f), Uniform(r, -1.35f, 1.35f) };
                m_S.sphereUnit.push_back(u);
                m_S.sphereIn.push_back(Vector3Length(u) <= 1.0f);
            }
            // a Fibonacci sphere: even coverage of the surface
            int n = shellCount[q];
            for (int i = 0; i < n; i++) {
                float y = 1.0f - 2.0f * (i + 0.5f) / n, rad = sqrtf(fmaxf(0.0f, 1.0f - y * y)), phi = i * 2.39996323f;
                m_S.shellDirs.push_back({ rad * cosf(phi), y, rad * sinf(phi) });
            }
        }

        // ---- the score -------------------------------------------------------------------------------------------------

        // One number for both poses, lower is better:
        //   fit       open pose against the mesh: the mesh volume the design misses and the design volume outside the
        //             mesh (sampled wherever the design is), both relative to the mesh's volume, how far out that extra
        //             material reaches, and how far the mesh surface is from the design surface
        //   sphere    closed pose against a sphere of its own volume around its own centroid: the same overlap and
        //             surface measures, and how far closed material reaches outside the sphere
        //   total     (1 - bias) * fit + bias * sphere + collisions * weight + costs
        // Collisions are bodies overlapping each other in the open pose and in the closed pose (not on the way).
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
            int nV = (int)m_T.vol.size(), nS = (int)m_T.surf.size(), nO = (int)m_S.openUnit.size(), nC = (int)m_S.closedUnit.size();
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
            {
                double mass = 0.0, off = 0.0, far = 0.0, overlap = 0.0, depth = 0.0;
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
                }
                if (mass > 1e-6) {
                    double sampleVolume = (double)oext.x * oext.y * oext.z / nO;
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
            double mass = 0.0, overlapC = 0.0, depthC = 0.0;
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
            }
            if (mass > 1e-6) {
                sc.clashClosed = (float)(overlapC / mass);
                sc.collision += 0.5f * sc.clashClosed + 0.5f * (float)(depthC / mass / occCap);
            }
            double boxVolume = (double)cext.x * cext.y * cext.z;
            double volume = nC > 0 ? mass / nC * boxVolume : 0.0;
            if (mass < 1e-3 || volume <= 0.0) {
                sc.sphere = 1.0f;
            } else {
                sc.centre = Vector3Scale(centroid, (float)(1.0 / mass));
                sc.radius = m_P.sphereDiameter > 0.0f ? m_P.sphereDiameter * 0.5f : (float)cbrt(3.0 * volume / (4.0 * PI));
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
                diag->clashClosed.clear();
                for (int k = 0; k < 24 && !clashes.empty(); k++) {
                    int i = clashes[Pick(m_Rng, (int)clashes.size())];
                    bool isClosed = i >= cBase;
                    Vector3 p = isClosed ? inBox(clo, cext, m_S.closedUnit[i - cBase]) : inBox(olo, oext, m_S.openUnit[i - oBase]);
                    diag->clashes.push_back({ p, who[i] });
                    diag->clashOther.push_back(who2[i]);
                    diag->clashClosed.push_back(isClosed);
                }
            }

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
            sc.base = (1.0f - w) * sc.fit + w * sc.sphere + penalty;
            sc.total = sc.base + m_P.collisionWeight * sc.collision;
            sc.valid = true;
            return sc;
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

        // A new hinge, sized to the model. Half the time it folds the child towards the middle of the mesh (where the
        // sphere wants to be), otherwise about a random axis by a random angle.
        int Search::AttachHinge(Design& d, int parent, int child, Vector3 pivot, Vector3 childPoint) {
            if (!d.CanHinge(parent, child)) return -1;
            int h = d.AddHinge(parent, child);
            Bakugan::Hinge& hg = d.hinges[h];
            const float L = m_T.L;
            hg.pivot = pivot;
            hg.start = 0.0f;
            hg.hardware = false;     // the score ignores it; Finalize puts it back
            hg.radius = std::clamp(0.03f * L, 1.6f, 4.0f);
            hg.pinRadius = hg.radius * 0.4f;
            hg.length = std::clamp(0.2f * L, 6.0f, 30.0f);
            hg.leafLength = std::clamp(0.07f * L, 2.0f, 10.0f);
            hg.leafThickness = std::clamp(0.025f * L, 1.2f, 3.0f);
            hg.knuckles = 3;
            Vector3 out = Vector3Subtract(childPoint, pivot);
            if (Vector3Length(out) < 1e-4f) out = RandomUnit(m_Rng);
            out = Vector3Normalize(out);
            Vector3 in = Vector3Subtract(m_T.centre, pivot);
            Vector3 axis = Vector3CrossProduct(out, in);
            if (Chance(m_Rng, 0.5f) && Vector3Length(axis) > 1e-3f * Vector3Length(in)) {
                hg.axis = Vector3Normalize(axis);
                float cosang = Vector3DotProduct(out, Vector3Normalize(in));
                hg.end = acosf(std::clamp(cosang, -1.0f, 1.0f)) * Uniform(m_Rng, 0.6f, 1.0f);
            } else {
                Vector3 r = RandomUnit(m_Rng);
                r = Vector3Subtract(r, Vector3Scale(out, Vector3DotProduct(r, out)));
                hg.axis = Vector3Length(r) > 1e-3f ? Vector3Normalize(r) : Utils::AnyPerpendicular(out);
                hg.end = Uniform(m_Rng, -PI, PI);
            }
            return hg.id;
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
            f->size = Vector3Scale(f->size, Uniform(m_Rng, 0.95f, 1.15f));
            f->rounding = Chance(m_Rng, 0.5f) ? Uniform(m_Rng, 0.4f, 1.0f) : Uniform(m_Rng);
            return true;
        }

        // ---- moves -----------------------------------------------------------------------------------------------------

        bool Search::PerturbShape(Design& d, int s, float scale) {
            const float L = m_T.L;
            SuperFrustum& f = d.shapes[s].sf;
            float r = Uniform(m_Rng);
            if (r < 0.32f) {
                f.position = Vector3Add(f.position, Vector3Scale(RandomUnit(m_Rng), fabsf(Normal(m_Rng, 0.035f * L * scale))));
            } else if (r < 0.54f) {
                if (Chance(m_Rng, 0.5f)) {
                    int a = Pick(m_Rng, 3);
                    (&f.size.x)[a] *= expf(Normal(m_Rng, 0.18f * scale));
                } else {
                    float k = expf(Normal(m_Rng, 0.1f * scale));
                    f.size = Vector3Scale(f.size, k);
                }
            } else if (r < 0.70f) {
                f.rotation = Jiggle(f.rotation, 0.3f * scale, m_Rng);
            } else if (r < 0.80f) {
                // resize one side and shift by half as much: the opposite face stays put
                int a = Pick(m_Rng, 3);
                float before = (&f.size.x)[a];
                (&f.size.x)[a] = fmaxf(before + Normal(m_Rng, 0.04f * L * scale), 0.01f * L);
                float sign = Chance(m_Rng, 0.5f) ? 1.0f : -1.0f;
                f.position = Vector3Add(f.position, Vector3Scale(f.Axis(a), sign * ((&f.size.x)[a] - before) * 0.5f));
            } else {
                switch (Pick(m_Rng, 5)) {
                    case 0: f.rounding = std::clamp(f.rounding + Normal(m_Rng, 0.25f * scale), 0.0f, 1.0f); break;
                    case 1: f.taper = std::clamp(f.taper + Normal(m_Rng, 0.25f * scale), 0.0f, 1.0f); break;
                    case 2: f.bulge = std::clamp(f.bulge + Normal(m_Rng, 0.2f * scale), -1.0f, 1.0f); break;
                    case 3: f.dilation = fmaxf(0.0f, f.dilation + Normal(m_Rng, 0.01f * L * scale)); break;
                    case 4: f.onion = Chance(m_Rng, 0.7f) ? 0.0f : Uniform(m_Rng, 0.3f, 0.9f); break;
                }
            }
            return true;
        }

        bool Search::PerturbHinge(Design& d, int h, float scale) {
            const float L = m_T.L;
            Bakugan::Hinge& hg = d.hinges[h];
            float r = Uniform(m_Rng);
            if (r < 0.38f) {
                // the closed pose only depends on the angle modulo a full turn (collisions are ignored)
                hg.end = hg.start + WrapAngle(hg.end - hg.start + Normal(m_Rng, 0.35f * scale));
            } else if (r < 0.46f) {
                // a fresh angle, or a quarter / half turn
                float turn = Chance(m_Rng, 0.5f) ? Uniform(m_Rng, -PI, PI) : (float)(Pick(m_Rng, 4) - 1) * PI * 0.5f;
                if (turn == 0.0f) turn = PI;
                hg.end = hg.start + turn;
            } else if (r < 0.70f) {
                hg.pivot = Vector3Add(hg.pivot, Vector3Scale(RandomUnit(m_Rng), fabsf(Normal(m_Rng, 0.035f * L * scale))));
            } else if (r < 0.93f) {
                Quaternion q = QuaternionFromAxisAngle(RandomUnit(m_Rng), Normal(m_Rng, 0.3f * scale));
                hg.axis = Vector3Normalize(Vector3RotateByQuaternion(hg.axis, q));
            } else {
                // hang it from another body
                std::vector<int> bodies = d.Bodies();
                int b = d.groups[bodies[Pick(m_Rng, (int)bodies.size())]].id;
                if (b == hg.parent || !d.CanHinge(b, hg.child, h)) return false;
                hg.parent = b;
                hg.pivot = PivotBetween(d, b, hg.child, BodyHint(d, b), BodyHint(d, hg.child));
            }
            return true;
        }

        // Small changes to the focused shape / hinge (or a random one): the continuous part of the grammar.
        bool Search::Perturb(Design& d, const Focus& f, float scale) {
            int s = d.ShapeIndex(f.shape), h = d.HingeIndex(f.hinge);
            if (s >= 0 && h >= 0) return Chance(m_Rng, 0.6f) ? PerturbShape(d, s, scale) : PerturbHinge(d, h, scale);
            if (s >= 0) return PerturbShape(d, s, scale);
            if (h >= 0) return PerturbHinge(d, h, scale);
            float r = Uniform(m_Rng);
            if (!d.hinges.empty() && r < 0.24f) return PerturbHinge(d, Pick(m_Rng, (int)d.hinges.size()), scale);
            if (r < 0.31f) {
                // a group's blend (a body's own members, or a feature's)
                int g = Pick(m_Rng, (int)d.groups.size());
                if (d.groups.empty() || d.GroupShapes(d.groups[g].id).size() < 2) return false;
                d.groups[g].blend = fmaxf(0.0f, d.groups[g].blend + Normal(m_Rng, 0.012f * m_T.L * scale));
                if (Chance(m_Rng, 0.15f)) d.groups[g].blend = 0.0f;
                return true;
            }
            if (d.shapes.empty()) return false;
            return PerturbShape(d, Pick(m_Rng, (int)d.shapes.size()), scale);
        }

        // The discrete part of the grammar. False when the chosen change doesn't apply to this design.
        bool Search::Structural(Design& d, const Diagnostics& dg, Focus* focus) {
            enum Move { AddShape, AddCut, RemoveShape, ToggleOp, MoveShape, AddBody, SplitBody, RemoveBodyMove, MergeBodyMove,
                        AddFeature, DissolveFeature, ToggleFeature, AddHinge, RemoveHinge, RemoveStray, CarveClash, Count };
            float weight[Count] = { 3.0f, 1.2f, 1.6f, 0.6f, 0.8f, 1.2f, 1.2f, 0.4f, 0.8f, 0.5f, 0.4f, 0.3f, 1.0f, 0.2f, 1.0f, 1.2f };
            if (dg.excess.empty()) weight[RemoveStray] = 0.0f;
            if (dg.clashes.empty() || m_P.collisionWeight <= 0.0f) weight[CarveClash] = 0.0f;
            if (!m_P.allowCuts) weight[AddCut] = weight[ToggleOp] = weight[ToggleFeature] = weight[CarveClash] = 0.0f;
            if (!m_P.allowFeatures) weight[AddFeature] = weight[ToggleFeature] = 0.0f;
            float sum = 0.0f;
            for (float w : weight) sum += w;
            float r = Uniform(m_Rng) * sum;
            int move = 0;
            while (move < Count - 1 && r >= weight[move]) r -= weight[move++];

            std::vector<int> bodies = d.Bodies();
            int nBodies = (int)bodies.size();
            bool roomShape = (int)d.shapes.size() < m_P.maxShapes, roomBody = nBodies < m_P.maxBodies;
            switch (move) {
                case AddShape: {
                    if (!roomShape) return false;
                    SuperFrustum f;
                    int g;
                    if (!FitRegion(d, dg, false, &f, &g)) return false;
                    int target = d.groups[g].id;
                    // sometimes into one of the body's blended features
                    std::vector<int> feats;
                    for (const Group& gr : d.groups)
                        if (gr.parent >= 0 && gr.op == Op::Union && d.BodyOf(gr.id) == target) feats.push_back(gr.id);
                    if (!feats.empty() && Chance(m_Rng, 0.3f)) target = feats[Pick(m_Rng, (int)feats.size())];
                    int s = d.AddShape(f, target, "Shape");
                    focus->shape = d.shapes[s].id;
                    return true;
                }
                case AddCut: {
                    if (!roomShape) return false;
                    SuperFrustum f;
                    int g;
                    if (!FitRegion(d, dg, true, &f, &g)) return false;
                    int s = d.AddShape(f, d.groups[g].id, "Cut", Op::Cut);
                    focus->shape = d.shapes[s].id;
                    return true;
                }
                case RemoveShape: {
                    if (d.shapes.size() <= 1) return false;
                    int s = Pick(m_Rng, (int)d.shapes.size());
                    int body = d.ShapeBody(s);
                    if (d.AddsMaterial(s) && MaterialShapes(d, body) <= 1) return false;
                    d.RemoveShape(s);
                    return true;
                }
                case ToggleOp: {
                    if (d.shapes.empty()) return false;
                    int s = Pick(m_Rng, (int)d.shapes.size());
                    if (d.AddsMaterial(s) && MaterialShapes(d, d.ShapeBody(s)) <= 1) return false;
                    Shape& sh = d.shapes[s];
                    sh.op = sh.op == Op::Union ? Op::Cut : Op::Union;
                    sh.color = sh.op == Op::Cut ? Color{ 225, 90, 90, 255 } : d.groups[d.GroupIndex(sh.group)].color;
                    focus->shape = sh.id;
                    return true;
                }
                case MoveShape: {
                    if (nBodies < 2 || d.shapes.empty()) return false;
                    int s = Pick(m_Rng, (int)d.shapes.size());
                    int from = d.ShapeBody(s);
                    if (d.AddsMaterial(s) && MaterialShapes(d, from) <= 1) return false;
                    int to = d.groups[bodies[Pick(m_Rng, nBodies)]].id;
                    if (to == from) return false;
                    d.MoveShape(s, to);
                    if (d.shapes[s].op == Op::Union) d.shapes[s].color = d.groups[d.GroupIndex(to)].color;
                    focus->shape = d.shapes[s].id;
                    return true;
                }
                case AddBody: {
                    // a new piece where material is missing, hinged to the nearest body
                    if (!roomShape || !roomBody) return false;
                    SuperFrustum f;
                    int g;
                    if (!FitRegion(d, dg, false, &f, &g)) return false;
                    Vector3 at = f.position;
                    int parent = d.groups[g].id;
                    int nb = d.AddGroup("Body", -1);
                    int body = d.groups[nb].id;
                    int s = d.AddShape(f, body, "Shape");
                    Vector3 pivot = PivotBetween(d, parent, body, BodyHint(d, parent), at);
                    focus->shape = d.shapes[s].id;
                    focus->hinge = AttachHinge(d, parent, body, pivot, at);
                    return true;
                }
                case SplitBody: {
                    // one shape of a body becomes a body of its own, hinged to what is left
                    if (!roomBody) return false;
                    int b = d.groups[bodies[Pick(m_Rng, nBodies)]].id;
                    std::vector<int> mat;
                    for (int s : d.DirectShapes(b))
                        if (d.AddsMaterial(s)) mat.push_back(s);
                    if (mat.size() < 2) return false;
                    int s = mat[Pick(m_Rng, (int)mat.size())];
                    int nb = d.AddGroup("Body", -1);
                    int body = d.groups[nb].id;
                    d.MoveShape(s, body);
                    d.shapes[s].color = d.groups[nb].color;
                    Vector3 at = d.shapes[s].sf.position;
                    Vector3 pivot = PivotBetween(d, b, body, BodyHint(d, b), at);
                    focus->hinge = AttachHinge(d, b, body, pivot, at);
                    return focus->hinge >= 0;
                }
                case RemoveBodyMove: {
                    if (nBodies < 2) return false;
                    RemoveBody(d, d.groups[bodies[Pick(m_Rng, nBodies)]].id);
                    return true;
                }
                case MergeBodyMove: {
                    if (d.hinges.empty()) return false;
                    return MergeBody(d, d.hinges[Pick(m_Rng, (int)d.hinges.size())].child);
                }
                case AddFeature: {
                    // one or two shapes of a group become a blended feature
                    if (d.groups.empty()) return false;
                    int g = d.groups[Pick(m_Rng, (int)d.groups.size())].id;
                    std::vector<int> direct = d.DirectShapes(g);
                    if (direct.size() < 2) return false;
                    std::shuffle(direct.begin(), direct.end(), m_Rng);
                    std::vector<int> ids = { d.shapes[direct[0]].id };
                    if (Chance(m_Rng, 0.6f)) ids.push_back(d.shapes[direct[1]].id);
                    int f = d.GroupItems(ids, {});
                    int fi = d.GroupIndex(f);
                    if (fi < 0 || d.groups[fi].parent < 0) return false;
                    d.groups[fi].blend = Uniform(m_Rng, 0.01f, 0.06f) * m_T.L;
                    return true;
                }
                case DissolveFeature: {
                    std::vector<int> feats;
                    for (int g = 0; g < (int)d.groups.size(); g++)
                        if (d.groups[g].parent >= 0) feats.push_back(g);
                    if (feats.empty()) return false;
                    d.Ungroup(feats[Pick(m_Rng, (int)feats.size())]);
                    return true;
                }
                case ToggleFeature: {
                    std::vector<int> feats;
                    for (int g = 0; g < (int)d.groups.size(); g++)
                        if (d.groups[g].parent >= 0) feats.push_back(g);
                    if (feats.empty()) return false;
                    Group& gr = d.groups[feats[Pick(m_Rng, (int)feats.size())]];
                    gr.op = gr.op == Op::Union ? Op::Cut : Op::Union;
                    return MaterialShapes(d, d.BodyOf(gr.id)) > 0;
                }
                case AddHinge: {
                    // hang a loose body on the nearest body it can hang from
                    std::vector<int> roots = Roots(d);
                    if (roots.size() < 2) return false;
                    int child = roots[Pick(m_Rng, (int)roots.size())];
                    Vector3 at = BodyHint(d, child);
                    int g = NearestBody(d, at, child);
                    if (g < 0) return false;
                    int parent = d.groups[g].id;
                    if (!d.CanHinge(parent, child)) {
                        int other = d.groups[bodies[Pick(m_Rng, nBodies)]].id;
                        if (!d.CanHinge(other, child)) return false;
                        parent = other;
                    }
                    Vector3 pivot = PivotBetween(d, parent, child, BodyHint(d, parent), at);
                    focus->hinge = AttachHinge(d, parent, child, pivot, at);
                    return focus->hinge >= 0;
                }
                case RemoveHinge: {
                    if (d.hinges.empty()) return false;
                    d.RemoveHinge(Pick(m_Rng, (int)d.hinges.size()));
                    return true;
                }
                case RemoveStray: {
                    // the shape that put material off the mesh at a spot the score found
                    const Spot& sp = dg.excess[Pick(m_Rng, (int)dg.excess.size())];
                    if (sp.body < 0 || sp.body >= (int)d.groups.size()) return false;
                    int body = d.groups[sp.body].id;
                    Vector3 q = d.Transforms(0.0f, false)[sp.body].ApplyInverse(sp.p);
                    int pick = -1;
                    float best = FLT_MAX;
                    for (int s : d.GroupShapes(body)) {
                        if (!d.AddsMaterial(s)) continue;
                        float dd = d.shapes[s].sf.Distance(q);
                        if (dd < best) { best = dd; pick = s; }
                    }
                    if (pick < 0 || MaterialShapes(d, body) <= 1) return false;
                    if (Chance(m_Rng, 0.5f)) {
                        d.RemoveShape(pick);
                    } else {
                        // or pull it back: shrink it towards the mesh
                        SuperFrustum& f = d.shapes[pick].sf;
                        Vector3 toward = Vector3Subtract(m_T.centre, f.position);
                        f.size = Vector3Scale(f.size, Uniform(m_Rng, 0.5f, 0.85f));
                        f.position = Vector3Add(f.position, Vector3Scale(toward, Uniform(m_Rng, 0.0f, 0.3f)));
                        focus->shape = d.shapes[pick].id;
                    }
                    return true;
                }
                case CarveClash: {
                    // a cut in one of two overlapping bodies, fitted to where they overlap
                    int k = Pick(m_Rng, (int)dg.clashes.size());
                    int ga = dg.clashes[k].body, gb = dg.clashOther[k];
                    if (ga < 0 || gb < 0 || ga >= (int)d.groups.size() || gb >= (int)d.groups.size()) return false;
                    if ((int)d.shapes.size() >= m_P.maxShapes) return false;
                    if (Chance(m_Rng, 0.5f)) std::swap(ga, gb);
                    std::vector<Rigid> xf = d.Transforms(dg.clashClosed[k] ? 1.0f : 0.0f, false);
                    BodyField fa = BodyField::Body(d, d.groups[ga].id), fb = BodyField::Body(d, d.groups[gb].id);
                    if (fa.Empty() || fb.Empty()) return false;
                    float rho = m_T.L * Uniform(m_Rng, 0.04f, 0.15f);
                    std::vector<Vector3> keep;
                    for (int i = 0; i < 300; i++) {
                        Vector3 p;
                        do p = { Uniform(m_Rng, -1, 1), Uniform(m_Rng, -1, 1), Uniform(m_Rng, -1, 1) };
                        while (Vector3LengthSqr(p) > 1.0f);
                        p = Vector3Add(dg.clashes[k].p, Vector3Scale(p, rho));
                        Vector3 qa = xf[ga].ApplyInverse(p);
                        if (fa.Distance(qa) < 0.0f && fb.Distance(xf[gb].ApplyInverse(p)) < 0.0f) keep.push_back(qa);
                    }
                    if (keep.size() < 10) return false;
                    SuperFrustum f = BoxAround(keep, 0.02f, 0.015f * m_T.L);
                    f.size = Vector3Scale(f.size, Uniform(m_Rng, 1.05f, 1.3f));
                    f.rounding = Uniform(m_Rng);
                    int s = d.AddShape(f, d.groups[ga].id, "Cut", Op::Cut);
                    focus->shape = d.shapes[s].id;
                    return true;
                }
            }
            return false;
        }

        bool Search::Propose(Design& d, const Diagnostics& dg, float progress, Focus* focus, bool* small) {
            float structural = 0.08f + 0.3f * (1.0f - progress);
            *small = !Chance(m_Rng, structural);
            if (!*small) return Structural(d, dg, focus);
            return Perturb(d, Focus{}, m_Step);
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

        void Search::Publish(const State& best, int iteration, bool force) {
            if (!m_Live) return;
            Clock::time_point now = Clock::now();
            if (!force && now - m_LastPublish < std::chrono::milliseconds(250)) return;
            m_LastPublish = now;
            Design d = best.design;
            Finalize(d);
            std::lock_guard<std::mutex> l(m_Live->mutex);
            m_Live->best = std::move(d);
            m_Live->score = best.score;
            m_Live->iteration = iteration;
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
                        Vector3 mid = (Vector3){ 0 };
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
            cur.score = Evaluate(cur.design, &cur.diag);
            if (!cur.score.valid) { res.message = "Could not score the starting design."; return res; }
            State best = cur;
            Publish(best, 0, true);

            const int N = std::max(0, m_P.iterations);
            // evenly spaced snapshots of the best design, for a recording of the run
            int recordCount = 0, recorded = 0;
            if (m_Live) {
                std::lock_guard<std::mutex> l(m_Live->mutex);
                recordCount = m_Live->recordFrames;
                m_Live->frames.clear();
            }
            auto record = [&](int iteration, bool last) {
                if (recordCount <= 0) return;
                while (recorded < recordCount &&
                       (last || (long long)iteration * std::max(1, recordCount - 1) >= (long long)recorded * std::max(1, N))) {
                    AutoFrame f;
                    f.design = best.design;
                    Finalize(f.design);
                    f.score = best.score;
                    f.iteration = iteration;
                    std::lock_guard<std::mutex> l(m_Live->mutex);
                    m_Live->frames.push_back(std::move(f));
                    recorded++;
                }
            };
            record(0, false);
            // collisions start cheap and reach their full weight at 60% of the run: hinges and folds get to form
            // before overlaps between the bodies have to be carved or moved away
            float ramp = 0.1f;
            auto objective = [&](const AutoScore& sc) { return sc.base + ramp * m_P.collisionWeight * sc.collision; };
            const float T0 = 0.004f, T1 = 0.00005f;
            int it = 0, accepted = 0, sinceBest = 0;
            int historyEvery = std::max(1, N / 240), nextHistory = historyEvery;
            if (m_Live) {
                std::lock_guard<std::mutex> l(m_Live->mutex);
                m_Live->iterations = N;
                m_Live->history.clear();
                m_Live->history.push_back(best.score.total);
            }
            m_C.SetStage("Searching");
            while (it < N && !m_C.Cancelled()) {
                float progress = (float)it / (float)std::max(1, N);
                float T = T0 * powf(T1 / T0, progress);
                ramp = std::min(1.0f, 0.1f + 0.9f * progress / 0.6f);
                State cand;
                cand.design = cur.design;
                Focus focus;
                bool ok = false, small = false;
                for (int tries = 0; tries < 8 && !ok; tries++) {
                    cand.design = cur.design;
                    focus = Focus{};
                    m_Score = cur.score;
                    ok = Propose(cand.design, cur.diag, progress, &focus, &small) && Sanitize(cand.design);
                }
                if (!ok) { it++; continue; }
                cand.score = Evaluate(cand.design, &cand.diag);
                it++;
                // a new shape or hinge rarely lands well: let it settle a few steps before it is judged
                if (focus.shape >= 0 || focus.hinge >= 0) {
                    for (int k = 0; k < 6 && it < N; k++) {
                        State t;
                        t.design = cand.design;
                        if (!Perturb(t.design, focus, fmaxf(m_Step, 0.3f)) || !Sanitize(t.design)) continue;
                        t.score = Evaluate(t.design, &t.diag);
                        it++;
                        if (t.score.valid && objective(t.score) < objective(cand.score)) cand = std::move(t);
                    }
                }
                if (!cand.score.valid) continue;
                // about a quarter of small changes should help: bigger steps while more do, smaller while fewer do
                if (small) m_Step = std::clamp(m_Step * (objective(cand.score) < objective(cur.score) ? 1.03f : 0.99f), 0.05f, 2.0f);
                float delta = objective(cand.score) - objective(cur.score);
                if (delta < 0.0f || Uniform(m_Rng) < expf(-delta / T)) {
                    cur = std::move(cand);
                    accepted++;
                    // the best is judged by the weights of the moment (fully weighted from 60% of the run on)
                    if (objective(cur.score) < objective(best.score) - 1e-6f) {
                        best = cur;
                        sinceBest = 0;
                        Publish(best, it, false);
                    }
                }
                // wandered off: go back to the best
                if (++sinceBest > 400 && objective(cur.score) > objective(best.score) + 0.01f) {
                    cur = best;
                    sinceBest = 0;
                }
                m_C.SetProgress(progress);
                record(it, false);
                if (m_Live && it >= nextHistory) {
                    nextHistory = it + historyEvery;
                    std::lock_guard<std::mutex> l(m_Live->mutex);
                    m_Live->history.push_back(best.score.total);
                    m_Live->iteration = it;
                    m_Live->accepted = accepted;
                    m_Live->temperature = T;
                }
            }
            m_C.SetProgress(1.0f);
            if (!m_C.Cancelled()) record(it, true);
            Publish(best, it, true);
            res.ok = true;
            res.cancelled = m_C.Cancelled();
            res.design = best.design;
            Finalize(res.design);
            res.score = best.score;
            const AutoScore& s = best.score;
            res.message = Utils::Format("%s %d bodies, %d shapes, %d hinges: covers %.0f%% of the mesh, %.0f%% outside it, sphere %.0f%%, "
                                        "overlapping %.1f%% open / %.1f%% closed (score %.3f, %d evaluations).",
                                        res.cancelled ? "Stopped, kept the best so far:" : (N == 0 ? "Scored:" : "Generated"),
                                        (int)res.design.Bodies().size(), (int)res.design.shapes.size(), (int)res.design.hinges.size(),
                                        s.fitCover * 100.0f, s.fitOutside * 100.0f, s.sphereIoU * 100.0f, s.clashOpen * 100.0f,
                                        s.clashClosed * 100.0f, s.total, m_Evaluations);
            return res;
        }
    }

    AutoResult AutoDesign(const TriMesh& mesh, const Design& start, const AutoParams& p, JobControl& c, std::shared_ptr<AutoLive> live) {
        Search s(mesh, p, c, std::move(live));
        return s.Run(start);
    }
}
