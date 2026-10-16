#include "utils.hpp"
#include <raymath.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>

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

    static void AddPiece(TriMesh& out, Mesh m, Vector3 scale, Vector3 rotDeg, Vector3 pos) {
        Matrix xf = MatrixMultiply(MatrixMultiply(MatrixScale(scale.x, scale.y, scale.z),
                                                  MatrixRotateXYZ(Vector3Scale(rotDeg, DEG2RAD))),
                                   MatrixTranslate(pos.x, pos.y, pos.z));
        AppendMesh(out, m, xf);
        UnloadMesh(m);
    }

    // A chunky four legged creature built from primitives.
    static TriMesh MakeCritter() {
        TriMesh t;
        AddPiece(t, GenMeshCube(1.3f, 0.65f, 0.75f), { 1, 1, 1 }, { 0, 0, 0 }, { 0, 0.95f, 0 });                  // torso
        AddPiece(t, GenMeshSphere(0.42f, 20, 20), { 1, 1, 1 }, { 0, 0, 0 }, { 0.9f, 1.45f, 0 });                   // head
        AddPiece(t, GenMeshCube(0.35f, 0.22f, 0.3f), { 1, 1, 1 }, { 0, 0, 0 }, { 1.32f, 1.36f, 0 });               // snout
        AddPiece(t, GenMeshCone(0.12f, 0.35f, 12), { 1, 1, 1 }, { 0, 0, -15 }, { 0.85f, 1.78f, 0.2f });            // ears
        AddPiece(t, GenMeshCone(0.12f, 0.35f, 12), { 1, 1, 1 }, { 0, 0, -15 }, { 0.85f, 1.78f, -0.2f });
        for (int sx = -1; sx <= 1; sx += 2)
            for (int sz = -1; sz <= 1; sz += 2)
                AddPiece(t, GenMeshCylinder(0.14f, 0.75f, 16), { 1, 1, 1 }, { 0, 0, 0 }, { 0.45f * sx, 0.0f, 0.24f * sz });
        AddPiece(t, GenMeshCylinder(0.09f, 0.8f, 12), { 1, 1, 1 }, { 0, 0, 55 }, { -0.6f, 1.05f, 0 });             // tail
        return t;
    }

    // A small winged dragon.
    static TriMesh MakeDragon() {
        TriMesh t;
        AddPiece(t, GenMeshSphere(0.5f, 24, 24), { 1.4f, 0.95f, 0.9f }, { 0, 0, 0 }, { 0, 1.0f, 0 });            // body
        AddPiece(t, GenMeshCylinder(0.16f, 0.7f, 14), { 1, 1, 1 }, { 0, 0, -35 }, { 0.5f, 1.2f, 0 });            // neck
        AddPiece(t, GenMeshSphere(0.25f, 18, 18), { 1.5f, 0.9f, 0.9f }, { 0, 0, 0 }, { 0.95f, 1.85f, 0 });       // head
        AddPiece(t, GenMeshCone(0.07f, 0.3f, 10), { 1, 1, 1 }, { 0, 0, 30 }, { 0.85f, 2.02f, 0.12f });          // horns
        AddPiece(t, GenMeshCone(0.07f, 0.3f, 10), { 1, 1, 1 }, { 0, 0, 30 }, { 0.85f, 2.02f, -0.12f });
        AddPiece(t, GenMeshCube(0.7f, 0.06f, 1.1f), { 1, 1, 1 }, { 25, 0, 0 }, { -0.05f, 1.45f, 0.85f });      // wings
        AddPiece(t, GenMeshCube(0.7f, 0.06f, 1.1f), { 1, 1, 1 }, { -25, 0, 0 }, { -0.05f, 1.45f, -0.85f });
        AddPiece(t, GenMeshCylinder(0.15f, 0.6f, 14), { 1, 1, 1 }, { 0, 0, 70 }, { -0.55f, 0.95f, 0 });          // tail
        AddPiece(t, GenMeshCylinder(0.1f, 0.55f, 12), { 1, 1, 1 }, { 0, 0, 100 }, { -1.1f, 0.75f, 0 });
        AddPiece(t, GenMeshCone(0.12f, 0.3f, 10), { 1, 1, 1 }, { 0, 0, 100 }, { -1.6f, 0.68f, 0 });
        for (int sx = -1; sx <= 1; sx += 2)
            for (int sz = -1; sz <= 1; sz += 2)
                AddPiece(t, GenMeshCylinder(0.12f, 0.65f, 14), { 1, 1, 1 }, { 0, 0, 0 }, { 0.32f * sx, 0.0f, 0.28f * sz });
        return t;
    }

    bool Utils::LoadAsset(const Asset& a, Model* out, std::string* err) {
        Mesh mesh = {};
        switch (a.primitive) {
            case Primitive::Cube:     mesh = GenMeshCube(1, 1, 1); break;
            case Primitive::Sphere:   mesh = GenMeshSphere(0.5f, 24, 24); break;
            case Primitive::Cylinder: mesh = GenMeshCylinder(0.3f, 1.5f, 24); break;
            case Primitive::Torus:    mesh = GenMeshTorus(0.3f, 1.0f, 24, 32); break;
            case Primitive::Knot:     mesh = GenMeshKnot(0.5f, 1.5f, 64, 128); break;
            case Primitive::Critter:
            case Primitive::Dragon: {
                TriMesh t = a.primitive == Primitive::Critter ? MakeCritter() : MakeDragon();
                ComputeNormals(t);
                *out = TriMeshToModel(t, Shader{});
                return true;
            }
            case Primitive::None: {
                if (!FileExists(a.path.c_str())) { *err = "File not found: " + a.path; return false; }
                Model m = LoadModel(a.path.c_str());
                if (m.meshCount == 0 || !IsModelValid(m)) {
                    UnloadModel(m);
                    *err = "Could not load " + a.name;
                    return false;
                }
                *out = m;
                return true;
            }
        }
        *out = LoadModelFromMesh(mesh);
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
        for (int k = 0; k < dim[2]; k++)
            for (int j = 0; j < dim[1]; j++)
                for (int i = 0; i < dim[0]; i++)
                    field[((size_t)k * dim[1] + j) * dim[0] + i] =
                        f({ origin.x + i * voxel, origin.y + j * voxel, origin.z + k * voxel });
        return SurfaceNets(field, dim, origin, voxel);
    }

    std::vector<Vector3> Utils::FibonacciSphere(int count) {
        std::vector<Vector3> dirs(count);
        const float golden = PI * (3.0f - sqrtf(5.0f));
        for (int i = 0; i < count; i++) {
            float y = 1.0f - 2.0f * (i + 0.5f) / count;
            float r = sqrtf(fmaxf(0.0f, 1.0f - y * y));
            float a = golden * i;
            dirs[i] = { r * cosf(a), y, r * sinf(a) };
        }
        return dirs;
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

    bool Utils::WriteOBJ(const std::string& path, const std::vector<const TriMesh*>& meshes, const std::vector<Rigid>& xf,
                         const std::vector<std::string>& names, std::string* err) {
        FILE* fp = fopen(path.c_str(), "w");
        if (!fp) { *err = "Could not write " + path; return false; }
        fprintf(fp, "# Bakugan Everything export\n");
        size_t base = 1;
        for (size_t m = 0; m < meshes.size(); m++) {
            const TriMesh& t = *meshes[m];
            fprintf(fp, "o %s\n", names[m].c_str());
            for (const Vector3& p : t.positions) {
                Vector3 w = xf[m].Apply(p);
                fprintf(fp, "v %.6f %.6f %.6f\n", w.x, w.y, w.z);
            }
            for (const Vector3& n : t.normals) {
                Vector3 w = xf[m].ApplyVector(n);
                fprintf(fp, "vn %.5f %.5f %.5f\n", w.x, w.y, w.z);
            }
            bool normals = t.normals.size() == t.positions.size();
            for (size_t i = 0; i + 2 < t.indices.size(); i += 3) {
                size_t a = base + t.indices[i], b = base + t.indices[i + 1], c = base + t.indices[i + 2];
                if (normals) fprintf(fp, "f %zu//%zu %zu//%zu %zu//%zu\n", a, a, b, b, c, c);
                else fprintf(fp, "f %zu %zu %zu\n", a, b, c);
            }
            base += t.positions.size();
        }
        fclose(fp);
        return true;
    }
}
