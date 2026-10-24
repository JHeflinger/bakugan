#include "raymarch.hpp"
#include <raymath.h>
#include <rlgl.h>
#include <algorithm>
#include <cmath>

namespace Bakugan {
    // ---- GLSL --------------------------------------------------------------------------------------------------------

    // Expects `uniform vec4 U[]` above it. Each function matches its C++ twin; keep them in step.
    static const char* Library = R"(
const float PI = 3.14159265358979;

// Vector3RotateByQuaternion
vec3 qrot(vec4 q, vec3 v) {
    vec3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// superfrustum.cpp: Segment / ConvexQuad
float segD2(vec2 p, vec2 a, vec2 b, out float side) {
    vec2 e = b - a, w = p - a;
    float t = clamp(dot(w, e) / (dot(e, e) + 1e-20), 0.0, 1.0);
    vec2 d = w - e * t;
    side = e.x * w.y - e.y * w.x;
    return dot(d, d);
}

float convexQuad(vec2 p, vec2 a, vec2 b, vec2 c, vec2 d) {
    float s0, s1, s2, s3;
    float best = min(min(segD2(p, a, b, s0), segD2(p, b, c, s1)), min(segD2(p, c, d, s2), segD2(p, d, a, s3)));
    bool inside = s0 >= 0.0 && s1 >= 0.0 && s2 >= 0.0 && s3 >= 0.0;
    float dist = sqrt(best);
    return inside ? -dist : dist;
}

// superfrustum.cpp: BendMap
void bendMap(inout float x, inout float z, float h, float bulge) {
    float s = bulge < 0.0 ? -1.0 : 1.0;
    x *= s;
    float theta = max(min(abs(bulge), 1.0) * (PI * 0.5), 1e-4);
    float st = sin(theta), ct = cos(theta);
    float c = h * ct / st;
    float rho = sqrt(c * c + h * h);
    float phi = atan(z, c - x);
    float nx, nz;
    if (phi > theta) {
        float dx = x, dz = z - h;
        nz = h + dx * st + dz * ct;
        nx = dx * ct - dz * st;
    } else if (phi < -theta) {
        float dx = x, dz = z + h;
        nz = -h - (dx * st - dz * ct);
        nx = dx * ct + dz * st;
    } else {
        float vx = x - c;
        nz = phi / theta * h;
        nx = rho - sqrt(vx * vx + z * z);
    }
    x = nx * s;
    z = nz;
}

// SuperFrustum::LocalDistance
float sfLocal(vec3 p, vec3 size, float rounding, float taper, float bulge, float dilation, float onion) {
    float hx = size.x * 0.5, hy = size.y * 0.5, h = size.z * 0.5;
    float x = p.x, z = p.z;
    if (abs(bulge) > 1e-4) bendMap(x, z, h, bulge);
    float inner = min(hx, hy);
    float r = rounding * inner;
    float qx = abs(x) - hx + r, qy = abs(p.y) - hy + r;
    float mx = max(qx, 0.0), my = max(qy, 0.0);
    float u = sqrt(mx * mx + my * my) + min(max(qx, qy), 0.0) - r;
    float x3 = -(1.0 - taper) * inner;
    float leftBottom, leftTop;
    if (onion > 1e-3) {
        leftBottom = -inner * (1.0 - onion);
        leftTop = -inner + (x3 + inner) * onion;
    } else {
        float far = -inner - 4.0 * (h + inner) - 1.0;
        leftBottom = far;
        leftTop = far;
    }
    return convexQuad(vec2(u, z), vec2(leftBottom, -h), vec2(0.0, -h), vec2(x3, h), vec2(leftTop, h)) - dilation;
}

// SuperFrustum::Distance, parameters at U[i..i+3]: position (w: cull radius), rotation, size + rounding, the rest
float sf(vec3 p, int i) {
    vec4 q = U[i + 1];
    vec3 l = qrot(vec4(-q.xyz, q.w), p - U[i].xyz);
    return sfLocal(l, U[i + 2].xyz, U[i + 2].w, U[i + 3].x, U[i + 3].y, U[i + 3].z, U[i + 3].w);
}

// field.cpp: SMin, and the union of a group (smooth when it has a blend)
float opU(float a, float b, float k) {
    if (k <= 0.0) return min(a, b);
    float h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}

// field.cpp: Cylinder (about z, radius r, z in [lo, hi]) and Box
float sdCyl(vec3 f, float r, float lo, float hi) {
    float dr = length(f.xy) - r;
    float dz = max(lo - f.z, f.z - hi);
    float ox = max(dr, 0.0), oz = max(dz, 0.0);
    return min(max(dr, dz), 0.0) + sqrt(ox * ox + oz * oz);
}

float sdBox(vec3 f, vec3 lo, vec3 hi) {
    vec3 q = abs(f - (lo + hi) * 0.5) - (hi - lo) * 0.5;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// field.cpp: ToFrame, hinge slots at U[b..]: [1] pivot, [2] leaf, [3] side, [4] axis
vec3 hwFrame(vec3 p, int b) {
    vec3 q = p - U[b + 1].xyz;
    return vec3(dot(q, U[b + 2].xyz), dot(q, U[b + 3].xyz), dot(q, U[b + 4].xyz));
}
)";

    static const char* Vertex = R"(#version 330
in vec3 vertexPosition;
uniform mat4 mvp;
uniform vec3 boxMin;
uniform vec3 boxMax;
out vec3 fragLocal;
void main() {
    fragLocal = boxMin + (vertexPosition + 0.5) * (boxMax - boxMin);
    gl_Position = mvp * vec4(vertexPosition, 1.0);
}
)";

    // Rays start at the camera (or, orthographic, far behind the pixel), are clipped to the box, and sphere trace the
    // field in the body's own (design pose) frame. Superfrustum distances are not exact everywhere, so steps are 0.9
    // of the distance. The hit writes its own depth and is lit like the meshes (shaders in utils.cpp).
    static const char* Main = R"(
in vec3 fragLocal;
uniform mat4 mvp;
uniform vec3 camLocal;
uniform vec3 camDir;
uniform int ortho;
uniform vec3 boxMin;
uniform vec3 boxMax;
uniform mat4 bodyRot;
uniform vec4 color;
uniform float highlight;
uniform float pixel;
out vec4 finalColor;

void main() {
    vec3 rd, ro;
    if (ortho == 1) { rd = normalize(camDir); ro = fragLocal - rd * 1.0e4; }
    else { ro = camLocal; rd = normalize(fragLocal - camLocal); }
    vec3 safe = vec3(abs(rd.x) > 1e-9 ? rd.x : 1e-9, abs(rd.y) > 1e-9 ? rd.y : 1e-9, abs(rd.z) > 1e-9 ? rd.z : 1e-9);
    vec3 inv = 1.0 / safe;
    vec3 t0 = (boxMin - ro) * inv, t1 = (boxMax - ro) * inv;
    vec3 tlo = min(t0, t1), thi = max(t0, t1);
    float tn = max(max(tlo.x, tlo.y), max(tlo.z, 0.0));
    float tf = min(min(thi.x, thi.y), thi.z);
    if (tn > tf) discard;
    float t = tn;
    bool hit = false;
    float eps = 0.002;
    for (int i = 0; i < 256; i++) {
        float d = map(ro + rd * t);
        eps = max(0.002, ortho == 1 ? pixel * 0.5 : t * pixel * 0.5);
        if (d < eps) { hit = true; break; }
        t += max(d * 0.9, eps * 0.5);
        if (t > tf) break;
    }
    if (!hit) discard;
    vec3 p = ro + rd * t;
    float e = max(eps, 0.003);
    vec2 k = vec2(1.0, -1.0);
    vec3 n = k.xyy * map(p + k.xyy * e) + k.yyx * map(p + k.yyx * e) + k.yxy * map(p + k.yxy * e) + k.xxx * map(p + k.xxx * e);
    n = length(n) > 1e-12 ? normalize(n) : -rd;
    vec3 nw = normalize(mat3(bodyRot) * n);
    vec3 vw = normalize(mat3(bodyRot) * -rd);
    vec3 key = normalize(vec3(0.4, 1.0, 0.6));
    if (dot(nw, vw) < 0.0) nw = -nw;
    float rim = pow(1.0 - max(dot(nw, vw), 0.0), 3.0);
    float light = 0.35 + 0.65 * max(dot(nw, key), 0.0) + 0.15 * rim;
    vec3 c = color.rgb * light + vec3(highlight * (0.25 + 0.75 * rim) * 0.55);
    finalColor = vec4(c, color.a);
    vec4 clip = mvp * vec4((p - boxMin) / (boxMax - boxMin) - 0.5, 1.0);
    gl_FragDepth = clamp(clip.z / clip.w * 0.5 + 0.5, 0.0, 1.0);
}
)";

    static const char* ProbeVertex = R"(#version 330
in vec3 vertexPosition;
void main() { gl_Position = vec4(vertexPosition.xy * 2.0, 0.0, 1.0); }
)";

    static const char* ProbeMain = R"(
uniform sampler2D points;
out vec4 finalColor;
void main() {
    vec3 p = texelFetch(points, ivec2(gl_FragCoord.xy), 0).xyz;
    finalColor = vec4(map(p), 0.0, 0.0, 1.0);
}
)";

    static std::string Assemble(const std::string& body, int uniforms, const char* main) {
        std::string s = "#version 330\n";
        s += "uniform vec4 U[" + std::to_string(std::max(uniforms, 1)) + "];\n";
        s += Library;
        s += "\nfloat map(vec3 p) {\n" + body + "}\n";
        s += main;
        return s;
    }

    std::string Sdf::FragmentSource(const std::string& body, int uniforms) { return Assemble(body, uniforms, Main); }
    const char* Sdf::VertexSource() { return Vertex; }
    std::string Sdf::ProbeFragmentSource(const std::string& body, int uniforms) { return Assemble(body, uniforms, ProbeMain); }
    const char* Sdf::ProbeVertexSource() { return ProbeVertex; }
    std::string Sdf::ShapeMap() { return "return sf(p, 0);\n"; }

    void Sdf::ShapeParams(const SuperFrustum& f, std::vector<Vector4>* U) {
        U->clear();
        U->push_back({ f.position.x, f.position.y, f.position.z, 0.0f });
        U->push_back({ f.rotation.x, f.rotation.y, f.rotation.z, f.rotation.w });
        U->push_back({ f.size.x, f.size.y, f.size.z, f.rounding });
        U->push_back({ f.taper, f.bulge, f.dilation, f.onion });
    }

    // ---- renderer ----------------------------------------------------------------------------------------------------

    bool SdfRenderer::Compile(Program& p, const std::string& code, int uniforms) {
        if (p.ok) UnloadShader(p.shader);
        p.ok = false;
        p.code = code;
        std::string fs = Sdf::FragmentSource(code, uniforms);
        p.shader = LoadShaderFromMemory(Vertex, fs.c_str());
        m_Compiles++;
        if (p.shader.id == 0 || p.shader.id == rlGetShaderIdDefault()) return false;
        p.ok = true;
        p.locU = GetShaderLocation(p.shader, "U");
        p.locCam = GetShaderLocation(p.shader, "camLocal");
        p.locDir = GetShaderLocation(p.shader, "camDir");
        p.locOrtho = GetShaderLocation(p.shader, "ortho");
        p.locMin = GetShaderLocation(p.shader, "boxMin");
        p.locMax = GetShaderLocation(p.shader, "boxMax");
        p.locRot = GetShaderLocation(p.shader, "bodyRot");
        p.locColor = GetShaderLocation(p.shader, "color");
        p.locHighlight = GetShaderLocation(p.shader, "highlight");
        p.locPixel = GetShaderLocation(p.shader, "pixel");
        return true;
    }

    bool SdfRenderer::Init() {
        m_Cube = GenMeshCube(1.0f, 1.0f, 1.0f);
        m_Material = LoadMaterialDefault();
        std::vector<Vector4> u;
        Sdf::ShapeParams(SuperFrustum{}, &u);
        m_Ok = Compile(m_Shape, Sdf::ShapeMap(), (int)u.size());
        return m_Ok;
    }

    void SdfRenderer::Unload() {
        for (auto& kv : m_Bodies)
            if (kv.second.ok) UnloadShader(kv.second.shader);
        m_Bodies.clear();
        if (m_Shape.ok) UnloadShader(m_Shape.shader);
        m_Shape.ok = false;
        // ours are unloaded above: hand the material the default shader so UnloadMaterial leaves it alone
        m_Material.shader.id = rlGetShaderIdDefault();
        m_Material.shader.locs = rlGetShaderLocsDefault();
        UnloadMaterial(m_Material);
        UnloadMesh(m_Cube);
        m_Ok = false;
    }

    bool SdfRenderer::SetBody(const std::string& slot, const BodyField& f) {
        Program& p = m_Bodies[slot];
        p.params.clear();
        std::string code = f.Glsl(&p.params);
        p.box = f.bounds;
        if (!p.ok || code != p.code) Compile(p, code, (int)p.params.size());
        return p.ok;
    }

    bool SdfRenderer::HasBody(const std::string& slot) const {
        auto it = m_Bodies.find(slot);
        return it != m_Bodies.end() && it->second.ok;
    }

    void SdfRenderer::Retain(const std::vector<std::string>& slots) {
        for (auto it = m_Bodies.begin(); it != m_Bodies.end();) {
            if (std::find(slots.begin(), slots.end(), it->first) == slots.end()) {
                if (it->second.ok) UnloadShader(it->second.shader);
                it = m_Bodies.erase(it);
            } else {
                ++it;
            }
        }
    }

    void SdfRenderer::Draw(Program& p, const Rigid& xf, Color color, float highlight, const SdfView& view) {
        if (!p.ok || p.params.empty()) return;
        Vector3 lo = p.box.min, hi = p.box.max;
        Vector3 size = Vector3Subtract(hi, lo);
        if (size.x <= 0.0f || size.y <= 0.0f || size.z <= 0.0f) return;
        Vector3 mid = Vector3Scale(Vector3Add(lo, hi), 0.5f);
        Matrix model = MatrixMultiply(MatrixMultiply(MatrixScale(size.x, size.y, size.z), MatrixTranslate(mid.x, mid.y, mid.z)), xf.ToMatrix());
        Rigid inv = xf.Inverse();
        Vector3 cam = inv.Apply(view.camera.position);
        Vector3 dir = inv.ApplyVector(Vector3Normalize(Vector3Subtract(view.camera.target, view.camera.position)));
        int ortho = view.ortho ? 1 : 0;
        float col[4] = { color.r / 255.0f, color.g / 255.0f, color.b / 255.0f, color.a / 255.0f };
        SetShaderValueV(p.shader, p.locU, p.params.data(), SHADER_UNIFORM_VEC4, (int)p.params.size());
        SetShaderValue(p.shader, p.locCam, &cam, SHADER_UNIFORM_VEC3);
        SetShaderValue(p.shader, p.locDir, &dir, SHADER_UNIFORM_VEC3);
        SetShaderValue(p.shader, p.locOrtho, &ortho, SHADER_UNIFORM_INT);
        SetShaderValue(p.shader, p.locMin, &lo, SHADER_UNIFORM_VEC3);
        SetShaderValue(p.shader, p.locMax, &hi, SHADER_UNIFORM_VEC3);
        SetShaderValueMatrix(p.shader, p.locRot, QuaternionToMatrix(xf.rotation));
        SetShaderValue(p.shader, p.locColor, col, SHADER_UNIFORM_VEC4);
        SetShaderValue(p.shader, p.locHighlight, &highlight, SHADER_UNIFORM_FLOAT);
        SetShaderValue(p.shader, p.locPixel, &view.pixel, SHADER_UNIFORM_FLOAT);
        m_Material.shader = p.shader;
        rlDrawRenderBatchActive();
        rlSetCullFace(RL_CULL_FACE_FRONT);   // back faces: one fragment per pixel, and it works with the camera inside
        DrawMesh(m_Cube, m_Material, model);
        rlDrawRenderBatchActive();
        rlSetCullFace(RL_CULL_FACE_BACK);
    }

    void SdfRenderer::DrawBody(const std::string& slot, const Rigid& xf, Color color, float highlight, const SdfView& view) {
        auto it = m_Bodies.find(slot);
        if (it != m_Bodies.end()) Draw(it->second, xf, color, highlight, view);
    }

    void SdfRenderer::DrawShape(const SuperFrustum& f, const Rigid& xf, Color color, float highlight, const SdfView& view) {
        Sdf::ShapeParams(f, &m_Shape.params);
        BoundingBox b = f.WorldBounds();
        float m = 0.05f + 0.01f * Vector3Distance(b.min, b.max);
        m_Shape.box = { Vector3SubtractValue(b.min, m), Vector3AddValue(b.max, m) };
        Draw(m_Shape, xf, color, highlight, view);
    }
}
