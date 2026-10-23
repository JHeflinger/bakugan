// TBB first: raylib's macros (PI, EPSILON, ...) must not leak into its headers.
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include "utils.hpp"
#include <raymath.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <array>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <cstdio>
#include <cstdlib>

namespace Bakugan {
    Vector3 Rigid::Apply(Vector3 p) const {
        return Vector3Add(Vector3RotateByQuaternion(p, rotation), position);
    }

    Vector3 Rigid::ApplyVector(Vector3 v) const {
        return Vector3RotateByQuaternion(v, rotation);
    }

    Vector3 Rigid::ApplyInverse(Vector3 p) const {
        return Vector3RotateByQuaternion(Vector3Subtract(p, position), { -rotation.x, -rotation.y, -rotation.z, rotation.w });
    }

    Rigid Rigid::Then(const Rigid& next) const {
        return { QuaternionNormalize(QuaternionMultiply(next.rotation, rotation)), next.Apply(position) };
    }

    Rigid Rigid::Inverse() const {
        Quaternion inv = { -rotation.x, -rotation.y, -rotation.z, rotation.w };
        return { inv, Vector3Negate(Vector3RotateByQuaternion(position, inv)) };
    }

    Matrix Rigid::ToMatrix() const {
        Matrix m = QuaternionToMatrix(rotation);
        m.m12 = position.x;
        m.m13 = position.y;
        m.m14 = position.z;
        return m;
    }

    Rigid Rigid::AboutAxis(Vector3 pivot, Vector3 axis, float angle) {
        Quaternion q = QuaternionFromAxisAngle(Vector3Normalize(axis), angle);
        // x -> q (x - pivot) + pivot
        return { q, Vector3Subtract(pivot, Vector3RotateByQuaternion(pivot, q)) };
    }

    const char* Utils::GetVertexShader() {
        return R"(#version 330
            in vec3 vertexPosition;
            in vec2 vertexTexCoord;
            in vec3 vertexNormal;
            in vec4 vertexColor;
            uniform mat4 mvp;
            uniform mat4 matModel;
            uniform mat4 matNormal;
            out vec3 fragPos;
            out vec2 fragTexCoord;
            out vec3 fragNormal;
            out vec4 fragColor;
            void main() {
                fragPos = vec3(matModel * vec4(vertexPosition, 1.0));
                fragTexCoord = vertexTexCoord;
                fragColor = vertexColor;
                fragNormal = vec3(matNormal * vec4(vertexNormal, 0.0));
                gl_Position = mvp * vec4(vertexPosition, 1.0);
            })";
    }

    const char* Utils::GetFragmentShader() {
        return R"(#version 330
            in vec3 fragPos;
            in vec2 fragTexCoord;
            in vec3 fragNormal;
            in vec4 fragColor;
            uniform sampler2D texture0;
            uniform vec4 colDiffuse;
            uniform vec3 viewPos;
            out vec4 finalColor;
            void main() {
                vec4 base = texture(texture0, fragTexCoord) * colDiffuse * fragColor;
                float light = 1.0;
                if (length(fragNormal) > 0.01) {
                    vec3 n = normalize(fragNormal);
                    vec3 v = normalize(viewPos - fragPos);
                    if (dot(n, v) < 0.0) n = -n;   // two-sided
                    vec3 key = normalize(vec3(0.4, 1.0, 0.6));
                    float rim = pow(1.0 - max(dot(n, v), 0.0), 3.0);
                    light = 0.35 + 0.65 * max(dot(n, key), 0.0) + 0.15 * rim;
                }
                finalColor = vec4(base.rgb * light, base.a);
            })";
    }

    bool Utils::IsModelFile(const char* path) {
        return IsFileExtension(path, ".obj;.gltf;.glb;.iqm;.vox;.m3d");
    }

    static void AppendMesh(TriMesh& out, const Mesh& mesh, Matrix xf) {
        if (!mesh.vertices) return;
        unsigned base = (unsigned)out.positions.size();
        for (int i = 0; i < mesh.vertexCount; i++)
            out.positions.push_back(Vector3Transform({ mesh.vertices[i * 3], mesh.vertices[i * 3 + 1], mesh.vertices[i * 3 + 2] }, xf));
        int tris = mesh.indices ? mesh.triangleCount : mesh.vertexCount / 3;
        for (int t = 0; t < tris; t++)
            for (int k = 0; k < 3; k++)
                out.indices.push_back(base + (mesh.indices ? mesh.indices[t * 3 + k] : (unsigned)(t * 3 + k)));
    }

    bool Utils::LoadModelFile(const std::string& path, Model* out, std::string* err) {
        if (!FileExists(path.c_str())) { *err = "File not found: " + path; return false; }
        if (IsFileExtension(path.c_str(), ".obj")) {
            // raylib's OBJ loader crashes on some files (e.g. without texture coordinates), so read it here
            TriMesh t;
            if (!ReadOBJ(path, &t, err)) return false;
            ComputeNormals(t);
            *out = TriMeshToModel(t, Shader{});
            return true;
        }
        Model m = LoadModel(path.c_str());
        if (m.meshCount == 0 || !IsModelValid(m)) {
            UnloadModel(m);
            *err = std::string("Could not load ") + GetFileName(path.c_str());
            return false;
        }
        *out = m;
        return true;
    }

    bool Utils::ReadOBJ(const std::string& path, TriMesh* out, std::string* err) {
        FILE* fp = fopen(path.c_str(), "r");
        if (!fp) { *err = "Could not open " + path; return false; }
        TriMesh t;
        char line[4096];
        std::vector<int> poly;
        while (fgets(line, sizeof(line), fp)) {
            const char* c = line;
            while (*c == ' ' || *c == '\t') c++;
            if (c[0] == 'v' && (c[1] == ' ' || c[1] == '\t')) {
                Vector3 v = { 0, 0, 0 };
                if (sscanf(c + 2, "%f %f %f", &v.x, &v.y, &v.z) >= 3) t.positions.push_back(v);
            } else if (c[0] == 'f' && (c[1] == ' ' || c[1] == '\t')) {
                poly.clear();
                const char* q = c + 2;
                while (*q) {
                    while (*q == ' ' || *q == '\t') q++;
                    if (!*q || *q == '\n' || *q == '\r') break;
                    char* end = nullptr;
                    long idx = strtol(q, &end, 10);
                    if (end == q) break;
                    int i = idx < 0 ? (int)t.positions.size() + (int)idx : (int)idx - 1;   // negative = relative
                    poly.push_back(i);
                    q = end;
                    while (*q && *q != ' ' && *q != '\t' && *q != '\n' && *q != '\r') q++;   // skip /vt/vn
                }
                for (size_t k = 2; k < poly.size(); k++) {
                    int a = poly[0], b = poly[k - 1], d = poly[k];
                    int n = (int)t.positions.size();
                    if (a < 0 || b < 0 || d < 0 || a >= n || b >= n || d >= n) continue;
                    t.indices.insert(t.indices.end(), { (unsigned)a, (unsigned)b, (unsigned)d });
                }
            }
        }
        fclose(fp);
        if (t.indices.empty()) { *err = "No faces found in " + path; return false; }
        ComputeBounds(t);
        *out = std::move(t);
        return true;
    }

    TriMesh Utils::ModelToTriMesh(const Model& m, Matrix xf) {
        TriMesh t;
        for (int i = 0; i < m.meshCount; i++) AppendMesh(t, m.meshes[i], xf);
        ComputeNormals(t);
        ComputeBounds(t);
        return t;
    }

    void Utils::ComputeNormals(TriMesh& mesh) {
        mesh.normals.assign(mesh.positions.size(), { 0, 0, 0 });
        for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
            unsigned a = mesh.indices[t], b = mesh.indices[t + 1], c = mesh.indices[t + 2];
            Vector3 n = Vector3CrossProduct(Vector3Subtract(mesh.positions[b], mesh.positions[a]),
                                            Vector3Subtract(mesh.positions[c], mesh.positions[a]));
            mesh.normals[a] = Vector3Add(mesh.normals[a], n);
            mesh.normals[b] = Vector3Add(mesh.normals[b], n);
            mesh.normals[c] = Vector3Add(mesh.normals[c], n);
        }
        for (Vector3& n : mesh.normals) n = Vector3Normalize(n);
    }

    void Utils::ComputeBounds(TriMesh& mesh) {
        Vector3 lo = { FLT_MAX, FLT_MAX, FLT_MAX }, hi = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
        for (const Vector3& p : mesh.positions) { lo = Vector3Min(lo, p); hi = Vector3Max(hi, p); }
        if (mesh.positions.empty()) lo = hi = { 0, 0, 0 };
        mesh.box = { lo, hi };
    }

    void Utils::CleanMesh(TriMesh& mesh) {
        struct Key { float x, y, z; bool operator==(const Key& o) const { return x == o.x && y == o.y && z == o.z; } };
        struct KeyHash {
            size_t operator()(const Key& k) const {
                uint32_t a, b, c;
                memcpy(&a, &k.x, 4); memcpy(&b, &k.y, 4); memcpy(&c, &k.z, 4);
                return ((size_t)a * 73856093u) ^ ((size_t)b * 19349663u) ^ ((size_t)c * 83492791u);
            }
        };
        std::unordered_map<Key, unsigned, KeyHash> weld;
        std::vector<Vector3> positions;
        std::vector<unsigned> remap(mesh.positions.size());
        for (size_t i = 0; i < mesh.positions.size(); i++) {
            const Vector3& p = mesh.positions[i];
            auto it = weld.emplace(Key{ p.x, p.y, p.z }, (unsigned)positions.size());
            if (it.second) positions.push_back(p);
            remap[i] = it.first->second;
        }
        struct Tri { unsigned a, b, c; };
        auto sorted = [](unsigned a, unsigned b, unsigned c) {
            unsigned lo = std::min(a, std::min(b, c)), hi = std::max(a, std::max(b, c));
            return std::array<unsigned, 3>{ lo, a + b + c - lo - hi, hi };
        };
        struct ArrHash {
            size_t operator()(const std::array<unsigned, 3>& k) const {
                return ((size_t)k[0] * 73856093u) ^ ((size_t)k[1] * 19349663u) ^ ((size_t)k[2] * 83492791u);
            }
        };
        std::vector<Tri> tris;
        std::unordered_map<std::array<unsigned, 3>, std::vector<size_t>, ArrHash> seen;
        for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
            unsigned a = remap[mesh.indices[t]], b = remap[mesh.indices[t + 1]], c = remap[mesh.indices[t + 2]];
            if (a == b || b == c || a == c) continue;
            Vector3 n = Vector3CrossProduct(Vector3Subtract(positions[b], positions[a]), Vector3Subtract(positions[c], positions[a]));
            if (Vector3LengthSqr(n) <= 0.0f) continue;
            seen[sorted(a, b, c)].push_back(tris.size());
            tris.push_back({ a, b, c });
        }
        // the same three corners more than once: keep one per orientation left after opposite pairs cancel
        std::vector<char> keep(tris.size(), 1);
        for (auto& kv : seen) {
            if (kv.second.size() < 2) continue;
            std::vector<size_t> fwd, rev;
            const Tri& r = tris[kv.second[0]];
            for (size_t i : kv.second) {
                const Tri& t = tris[i];
                bool same = (t.a == r.a && t.b == r.b) || (t.b == r.a && t.c == r.b) || (t.c == r.a && t.a == r.b);
                (same ? fwd : rev).push_back(i);
                keep[i] = 0;
            }
            int net = (int)fwd.size() - (int)rev.size();
            if (net > 0) keep[fwd[0]] = 1;
            if (net < 0) keep[rev[0]] = 1;
        }
        mesh.positions = std::move(positions);
        mesh.indices.clear();
        for (size_t i = 0; i < tris.size(); i++)
            if (keep[i]) mesh.indices.insert(mesh.indices.end(), { tris[i].a, tris[i].b, tris[i].c });
    }

    Model Utils::TriMeshToModel(const TriMesh& t, Shader lit) {
        // Unindexed, so there is no 16 bit index limit.
        Mesh mesh = {};
        int n = (int)t.indices.size();
        mesh.vertexCount = n;
        mesh.triangleCount = n / 3;
        mesh.vertices = (float*)MemAlloc(n * 3 * sizeof(float));
        mesh.normals = (float*)MemAlloc(n * 3 * sizeof(float));
        bool hasNormals = t.normals.size() == t.positions.size();
        for (int i = 0; i < n; i++) {
            Vector3 p = t.positions[t.indices[i]];
            Vector3 q = hasNormals ? t.normals[t.indices[i]] : Vector3{ 0, 0, 0 };
            mesh.vertices[i * 3] = p.x; mesh.vertices[i * 3 + 1] = p.y; mesh.vertices[i * 3 + 2] = p.z;
            mesh.normals[i * 3] = q.x; mesh.normals[i * 3 + 1] = q.y; mesh.normals[i * 3 + 2] = q.z;
        }
        if (n == 0) {
            // keep raylib happy with one degenerate triangle
            MemFree(mesh.vertices);
            MemFree(mesh.normals);
            mesh.vertexCount = 3;
            mesh.triangleCount = 1;
            mesh.vertices = (float*)MemAlloc(9 * sizeof(float));
            mesh.normals = (float*)MemAlloc(9 * sizeof(float));
        }
        UploadMesh(&mesh, false);
        Model m = LoadModelFromMesh(mesh);
        if (lit.id != 0)
            for (int i = 0; i < m.materialCount; i++) m.materials[i].shader = lit;
        return m;
    }

    TriMesh Utils::SurfaceNets(const std::vector<float>& f, const int dim[3], Vector3 origin, float voxel) {
        TriMesh out;
        const int nx = dim[0], ny = dim[1], nz = dim[2];
        if (nx < 2 || ny < 2 || nz < 2) return out;
        auto at = [&](int i, int j, int k) { return f[((size_t)k * ny + j) * nx + i]; };
        const int cx = nx - 1, cy = ny - 1, cz = nz - 1;
        std::vector<int> cell((size_t)cx * cy * cz, -1);
        auto cidx = [&](int i, int j, int k) { return ((size_t)k * cy + j) * cx + i; };
        static const int edges[12][2] = { { 0, 1 }, { 2, 3 }, { 4, 5 }, { 6, 7 }, { 0, 2 }, { 1, 3 },
                                          { 4, 6 }, { 5, 7 }, { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 } };
        for (int k = 0; k < cz; k++)
            for (int j = 0; j < cy; j++)
                for (int i = 0; i < cx; i++) {
                    float v[8];
                    int mask = 0;
                    for (int c = 0; c < 8; c++) {
                        v[c] = at(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1));
                        if (v[c] < 0) mask |= 1 << c;
                    }
                    if (mask == 0 || mask == 255) continue;
                    Vector3 sum = { 0, 0, 0 };
                    int count = 0;
                    for (const auto& e : edges) {
                        float a = v[e[0]], b = v[e[1]];
                        if ((a < 0) == (b < 0)) continue;
                        float t = a / (a - b);
                        Vector3 pa = { (float)(e[0] & 1), (float)((e[0] >> 1) & 1), (float)((e[0] >> 2) & 1) };
                        Vector3 pb = { (float)(e[1] & 1), (float)((e[1] >> 1) & 1), (float)((e[1] >> 2) & 1) };
                        sum = Vector3Add(sum, Vector3Lerp(pa, pb, t));
                        count++;
                    }
                    sum = Vector3Scale(sum, 1.0f / count);
                    cell[cidx(i, j, k)] = (int)out.positions.size();
                    out.positions.push_back({ origin.x + (i + sum.x) * voxel, origin.y + (j + sum.y) * voxel,
                                              origin.z + (k + sum.z) * voxel });
                }
        // one quad per sign-changing grid edge, joining the four cells around it
        for (int a = 0; a < 3; a++) {
            int b = (a + 1) % 3, c = (a + 2) % 3;
            for (int k = 0; k < nz; k++)
                for (int j = 0; j < ny; j++)
                    for (int i = 0; i < nx; i++) {
                        int p[3] = { i, j, k };
                        if (p[a] > dim[a] - 2 || p[b] < 1 || p[c] < 1 || p[b] > dim[b] - 2 || p[c] > dim[c] - 2) continue;
                        int q[3] = { i, j, k };
                        q[a] += 1;
                        float v0 = at(p[0], p[1], p[2]), v1 = at(q[0], q[1], q[2]);
                        if ((v0 < 0) == (v1 < 0)) continue;
                        int ids[4];
                        const int off[4][2] = { { -1, -1 }, { 0, -1 }, { 0, 0 }, { -1, 0 } };
                        bool ok = true;
                        for (int n = 0; n < 4; n++) {
                            int cc[3] = { p[0], p[1], p[2] };
                            cc[b] += off[n][0];
                            cc[c] += off[n][1];
                            if (cc[b] < 0 || cc[c] < 0 || cc[b] >= dim[b] - 1 || cc[c] >= dim[c] - 1) { ok = false; break; }
                            ids[n] = cell[cidx(cc[0], cc[1], cc[2])];
                            if (ids[n] < 0) { ok = false; break; }
                        }
                        if (!ok) continue;
                        if (v0 < 0) {
                            out.indices.insert(out.indices.end(), { (unsigned)ids[0], (unsigned)ids[1], (unsigned)ids[2],
                                                                    (unsigned)ids[0], (unsigned)ids[2], (unsigned)ids[3] });
                        } else {
                            out.indices.insert(out.indices.end(), { (unsigned)ids[0], (unsigned)ids[2], (unsigned)ids[1],
                                                                    (unsigned)ids[0], (unsigned)ids[3], (unsigned)ids[2] });
                        }
                    }
        }
        ComputeNormals(out);
        ComputeBounds(out);
        return out;
    }

    TriMesh Utils::MeshImplicit(const std::function<float(Vector3)>& f, BoundingBox box, int resolution) {
        Vector3 size = Vector3Subtract(box.max, box.min);
        float longest = fmaxf(size.x, fmaxf(size.y, size.z));
        float voxel = fmaxf(longest, 1e-6f) / (float)std::max(resolution, 4);
        Vector3 origin = Vector3Subtract(box.min, { voxel * 1.5f, voxel * 1.5f, voxel * 1.5f });
        int dim[3];
        for (int a = 0; a < 3; a++) dim[a] = (int)ceilf((&size.x)[a] / voxel) + 4;
        std::vector<float> field((size_t)dim[0] * dim[1] * dim[2]);
        // slices in parallel: this runs on the UI thread whenever a shape's size or profile changes
        tbb::parallel_for(tbb::blocked_range<int>(0, dim[2]), [&](const tbb::blocked_range<int>& r) {
            for (int k = r.begin(); k != r.end(); k++)
                for (int j = 0; j < dim[1]; j++)
                    for (int i = 0; i < dim[0]; i++)
                        field[((size_t)k * dim[1] + j) * dim[0] + i] =
                            f({ origin.x + i * voxel, origin.y + j * voxel, origin.z + k * voxel });
        });
        return SurfaceNets(field, dim, origin, voxel);
    }

    Color Utils::PartColor(int index) {
        float hue = fmodf(index * 137.508f + 20.0f, 360.0f);
        return ColorFromHSV(hue, 0.55f, 0.95f);
    }

    Vector3 Utils::QuaternionToEulerDegrees(Quaternion q) {
        return Vector3Scale(QuaternionToEuler(q), RAD2DEG);
    }

    Quaternion Utils::EulerDegreesToQuaternion(Vector3 d) {
        return QuaternionNormalize(QuaternionFromEuler(d.x * DEG2RAD, d.y * DEG2RAD, d.z * DEG2RAD));
    }

    Vector3 Utils::AnyPerpendicular(Vector3 v) {
        Vector3 a = fabsf(v.x) < 0.8f ? Vector3{ 1, 0, 0 } : Vector3{ 0, 1, 0 };
        return Vector3Normalize(Vector3CrossProduct(v, a));
    }

    bool Utils::WriteSTL(const std::string& path, const TriMesh& mesh, std::string* err) {
        FILE* fp = fopen(path.c_str(), "wb");
        if (!fp) { *err = "Could not write " + path; return false; }
        char header[80] = {};
        snprintf(header, sizeof(header), "Bakugan hinge designer export (mm)");
        fwrite(header, 1, 80, fp);
        uint32_t count = (uint32_t)(mesh.indices.size() / 3);
        fwrite(&count, 4, 1, fp);
        for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
            Vector3 a = mesh.positions[mesh.indices[t]], b = mesh.positions[mesh.indices[t + 1]], c = mesh.positions[mesh.indices[t + 2]];
            Vector3 n = Vector3Normalize(Vector3CrossProduct(Vector3Subtract(b, a), Vector3Subtract(c, a)));
            float rec[12] = { n.x, n.y, n.z, a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z };
            uint16_t attr = 0;
            fwrite(rec, sizeof(float), 12, fp);
            fwrite(&attr, 2, 1, fp);
        }
        bool ok = !ferror(fp);
        fclose(fp);
        if (!ok) *err = "Error while writing " + path;
        return ok;
    }
}
