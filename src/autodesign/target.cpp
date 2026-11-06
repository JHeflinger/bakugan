#include "target.hpp"
#include <algorithm>
#include <cfloat>
#include <cmath>

namespace Bakugan::Auto {
    // ---- inside test ------------------------------------------------------------------------------------------------

    InsideTest::InsideTest(const TriMesh& m, BoundingBox box) : m_Mesh(m) {
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

    bool InsideTest::Inside(Vector3 p) const {
        int votes = 0;
        for (const Axis& ax : m_Axes) votes += Crossings(ax, p) & 1;
        return votes >= 2;
    }

    int InsideTest::Crossings(const Axis& ax, Vector3 p) const {
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

    // ---- signed distance field ----------------------------------------------------------------------------------------

    // Exact Euclidean distance transform (Felzenszwalb & Huttenlocher) of one row of squared distances, in place.
    static void Edt1D(std::vector<double>& f, int n, std::vector<double>& d, std::vector<int>& v, std::vector<double>& z) {
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

    // Squared distance (in cells) from every grid point to the nearest point with `feature` set: the 1D transform
    // along x, then y, then z.
    static std::vector<double> Edt3D(const std::vector<char>& feature, const int n[3]) {
        std::vector<double> g(feature.size());
        for (size_t i = 0; i < g.size(); i++) g[i] = feature[i] ? 0.0 : 1e20;
        int big = std::max(n[0], std::max(n[1], n[2]));
        std::vector<double> f(big), d(big), z(big + 1);
        std::vector<int> v(big);
        size_t stride[3] = { 1, (size_t)n[0], (size_t)n[0] * n[1] };
        for (int a = 0; a < 3; a++) {
            int b = (a + 1) % 3, c = (a + 2) % 3;
            for (int j = 0; j < n[c]; j++)
                for (int i = 0; i < n[b]; i++) {
                    size_t base = i * stride[b] + j * stride[c];
                    for (int k = 0; k < n[a]; k++) f[k] = g[base + k * stride[a]];
                    Edt1D(f, n[a], d, v, z);
                    for (int k = 0; k < n[a]; k++) g[base + k * stride[a]] = f[k];
                }
        }
        return g;
    }

    void MeshField::Build(BoundingBox box, float pad, int res, const InsideTest& inside, Pool& pool) {
        box = Grow(box, pad);
        Vector3 ext = Vector3Subtract(box.max, box.min);
        m_H = MaxOf(ext) / res;
        m_Min = box.min;
        for (int a = 0; a < 3; a++) m_N[a] = std::max(2, (int)ceilf(Coord(ext, a) / m_H) + 1);
        size_t count = (size_t)m_N[0] * m_N[1] * m_N[2];
        std::vector<char> in(count, 0);
        pool.For(m_N[1] * m_N[2], [&](int r) {
            int y = r % m_N[1], z = r / m_N[1];
            for (int x = 0; x < m_N[0]; x++) in[((size_t)z * m_N[1] + y) * m_N[0] + x] = inside.Inside(Point(x, y, z));
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

    float MeshField::Distance(Vector3 p) const {
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

    // ---- the target ---------------------------------------------------------------------------------------------------

    bool Target::Build(const TriMesh& m, int quality, Rng& rng, Pool& pool, JobControl& c) {
        const int q = std::clamp(quality, 0, 2);
        const int surfCount[3] = { 600, 1200, 2400 };
        const int volCount[3] = { 2000, 4000, 8000 };
        const int fieldRes[3] = { 48, 72, 96 };
        mesh = &m;
        c.SetStage("Sampling the mesh");
        size_t tris = m.indices.size() / 3;
        if (tris == 0) return false;
        Vector3 lo = { FLT_MAX, FLT_MAX, FLT_MAX }, hi = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
        for (unsigned i : m.indices) { lo = Vector3Min(lo, m.positions[i]); hi = Vector3Max(hi, m.positions[i]); }
        box = { lo, hi };
        L = fmaxf(Vector3Distance(lo, hi), 1e-3f);

        // area weighted surface points
        std::vector<double> cdf(tris);
        std::vector<Vector3> normals(tris);
        double total = 0.0;
        for (size_t t = 0; t < tris; t++) {
            Vector3 a = m.positions[m.indices[t * 3]], b = m.positions[m.indices[t * 3 + 1]], cc = m.positions[m.indices[t * 3 + 2]];
            Vector3 n = Vector3CrossProduct(Vector3Subtract(b, a), Vector3Subtract(cc, a));
            float area = Vector3Length(n) * 0.5f;
            normals[t] = area > 0.0f ? Vector3Scale(n, 0.5f / area) : Vector3{ 0, 1, 0 };
            total += area;
            cdf[t] = total;
        }
        if (total <= 0.0) return false;
        auto surfacePoint = [&](Vector3* normal) {
            double r = Uniform(rng) * total;
            size_t t = std::min((size_t)(std::lower_bound(cdf.begin(), cdf.end(), r) - cdf.begin()), tris - 1);
            float u = Uniform(rng), v = Uniform(rng);
            if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
            Vector3 a = m.positions[m.indices[t * 3]], b = m.positions[m.indices[t * 3 + 1]], cc = m.positions[m.indices[t * 3 + 2]];
            if (normal) *normal = normals[t];
            return Vector3Add(a, Vector3Add(Vector3Scale(Vector3Subtract(b, a), u), Vector3Scale(Vector3Subtract(cc, a), v)));
        };
        for (int i = 0; i < surfCount[q]; i++) surf.push_back(surfacePoint(nullptr));

        // volume points: half uniform in a padded box, half scattered about the surface, where the fit is decided
        BoundingBox pad = Grow(box, 0.06f * L);
        Vector3 ext = Vector3Subtract(pad.max, pad.min);
        for (int i = 0; i < volCount[q]; i++) {
            if (i % 2 == 0) {
                vol.push_back({ pad.min.x + Uniform(rng) * ext.x, pad.min.y + Uniform(rng) * ext.y, pad.min.z + Uniform(rng) * ext.z });
            } else {
                Vector3 n;
                Vector3 s = surfacePoint(&n);
                vol.push_back(Vector3Add(s, Vector3Scale(n, Normal(rng, 0.03f * L))));
            }
        }
        if (c.Cancelled()) return false;

        c.SetStage("Finding the inside of the mesh");
        test = std::make_unique<InsideTest>(m, box);
        volIn.assign(vol.size(), 0);
        pool.ForEach((int)vol.size(), [&](int i) { volIn[i] = Inside(vol[i]); });
        for (size_t i = 0; i < vol.size(); i++)
            if (volIn[i]) inside.push_back(vol[i]);
        if (c.Cancelled()) return false;

        c.SetStage("Measuring distances to the mesh");
        field.Build(box, 0.2f * L, fieldRes[q], *test, pool);

        // more inside points, from the box
        for (int round = 0; round < 6 && inside.size() < 2000; round++) {
            std::vector<Vector3> cand(4000);
            for (Vector3& p : cand)
                p = { box.min.x + Uniform(rng) * (hi.x - lo.x), box.min.y + Uniform(rng) * (hi.y - lo.y), box.min.z + Uniform(rng) * (hi.z - lo.z) };
            std::vector<char> in(cand.size(), 0);
            pool.ForEach((int)cand.size(), [&](int i) { in[i] = Inside(cand[i]); });
            for (size_t i = 0; i < cand.size(); i++)
                if (in[i]) inside.push_back(cand[i]);
        }
        if (inside.size() < 20 || c.Cancelled()) return false;
        Vector3 sum = { 0, 0, 0 };
        for (Vector3 p : inside) sum = Vector3Add(sum, p);
        centre = Vector3Scale(sum, 1.0f / inside.size());
        return true;
    }
}
