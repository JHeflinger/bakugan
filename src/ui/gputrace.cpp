#include "gputrace.hpp"
#include "ui/raymarch.hpp"
#include <raymath.h>
#include <rlgl.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>

namespace Bakugan {
    // ---- the hand-off -----------------------------------------------------------------------------------------------------

    bool GpuSilhouettes::Trace(const TraceRequest& request, TraceResult* result) {
        std::unique_lock<std::mutex> l(m_Mutex);
        if (m_Closed || !m_Ok || m_Pending) return false;
        m_Request = &request;
        m_Result = result;
        m_Pending = true;
        m_Done = false;
        m_Asked.notify_all();
        // (the UI thread answers while it holds the lock, so this can't give up in the middle of an answer; it gives up
        // when the UI thread isn't getting to it at all: a dialog is open, the window is being dragged, ...)
        m_Answered.wait_for(l, std::chrono::seconds(2), [&] { return m_Done || m_Closed; });
        bool ok = m_Done && m_Success;
        m_Pending = false;
        m_Request = nullptr;
        m_Result = nullptr;
        return ok;
    }

    void GpuSilhouettes::Service(double seconds) {
        using Clock = std::chrono::steady_clock;
        const Clock::time_point until = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(std::max(0.0, seconds)));
        std::unique_lock<std::mutex> l(m_Mutex);
        for (;;) {
            if (m_Pending && !m_Done) {
                m_Success = Draw(*m_Request, m_Result);
                m_Done = true;
                m_LastAsked = GetTime();
                m_Answered.notify_all();
            }
            // stay for the next one only while a search is at it (it asked within the last quarter second)
            if (m_Closed || Clock::now() >= until || GetTime() - m_LastAsked > 0.25) break;
            m_Asked.wait_until(l, until);
        }
    }

    void GpuSilhouettes::Close() {
        std::lock_guard<std::mutex> l(m_Mutex);
        m_Closed = true;
        m_Answered.notify_all();
    }

    // ---- drawing ------------------------------------------------------------------------------------------------------------

    bool GpuSilhouettes::Target(int width, int height) {
        if (m_Target.id != 0 && m_Target.texture.width == width && m_Target.texture.height == height) return true;
        Drop();
        m_Target.id = rlLoadFramebuffer();
        if (m_Target.id == 0) return false;
        rlEnableFramebuffer(m_Target.id);
        m_Target.texture.id = rlLoadTexture(nullptr, width, height, RL_PIXELFORMAT_UNCOMPRESSED_R32G32B32A32, 1);
        m_Target.texture.width = width;
        m_Target.texture.height = height;
        m_Target.texture.format = PIXELFORMAT_UNCOMPRESSED_R32G32B32A32;
        m_Target.texture.mipmaps = 1;
        m_Target.depth.id = rlLoadTextureDepth(width, height, true);
        m_Target.depth.width = width;
        m_Target.depth.height = height;
        m_Target.depth.format = 19;
        m_Target.depth.mipmaps = 1;
        rlFramebufferAttach(m_Target.id, m_Target.texture.id, RL_ATTACHMENT_COLOR_CHANNEL0, RL_ATTACHMENT_TEXTURE2D, 0);
        rlFramebufferAttach(m_Target.id, m_Target.depth.id, RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_RENDERBUFFER, 0);
        bool complete = m_Target.texture.id != 0 && rlFramebufferComplete(m_Target.id);
        rlDisableFramebuffer();
        if (complete) {
            rlTextureParameters(m_Target.texture.id, RL_TEXTURE_MIN_FILTER, RL_TEXTURE_FILTER_NEAREST);
            rlTextureParameters(m_Target.texture.id, RL_TEXTURE_MAG_FILTER, RL_TEXTURE_FILTER_NEAREST);
        } else {
            Drop();
        }
        return complete;
    }

    void GpuSilhouettes::Drop() {
        if (m_Target.id == 0) return;
        if (m_Target.texture.id != 0) rlUnloadTexture(m_Target.texture.id);
        rlUnloadFramebuffer(m_Target.id);                // (and its depth buffer)
        m_Target = {};
    }

    bool GpuSilhouettes::Init() {
        m_Ok = Target(64, 64);
        if (!m_Ok) m_Error = "This OpenGL driver has no 32 bit float render targets: the silhouettes stay on the CPU.";
        return m_Ok;
    }

    void GpuSilhouettes::Unload() {
        Close();
        for (auto& kv : m_Programs)
            if (kv.second.ok) UnloadShader(kv.second.shader);
        m_Programs.clear();
        Drop();
        m_Ok = false;
    }

    GpuSilhouettes::Program* GpuSilhouettes::ProgramFor(const std::string& code, int uniforms) {
        std::string key = std::to_string(uniforms) + "\n" + code;
        auto it = m_Programs.find(key);
        if (it != m_Programs.end()) return &it->second;
        if (m_Programs.size() >= 256) {                  // a long search: start the cache over
            for (auto& kv : m_Programs)
                if (kv.second.ok) UnloadShader(kv.second.shader);
            m_Programs.clear();
        }
        Program p;
        std::string fs = Sdf::TraceFragmentSource(code, uniforms);
        p.shader = LoadShaderFromMemory(Sdf::TraceVertexSource(), fs.c_str());
        m_Compiles++;
        p.ok = p.shader.id != 0 && p.shader.id != rlGetShaderIdDefault();
        if (p.ok) {
            p.locU = GetShaderLocation(p.shader, "U");
            p.locDir = GetShaderLocation(p.shader, "viewDir");
            p.locViewU = GetShaderLocation(p.shader, "viewU");
            p.locViewV = GetShaderLocation(p.shader, "viewV");
            p.locCentre = GetShaderLocation(p.shader, "frameCentre");
            p.locHalf = GetShaderLocation(p.shader, "frameHalf");
            p.locRes = GetShaderLocation(p.shader, "viewRes");
            p.locTau = GetShaderLocation(p.shader, "tau");
            p.locFar = GetShaderLocation(p.shader, "farAway");
            p.locQuat = GetShaderLocation(p.shader, "bodyQuat");
            p.locPos = GetShaderLocation(p.shader, "bodyPos");
            p.locMin = GetShaderLocation(p.shader, "boxMin");
            p.locMax = GetShaderLocation(p.shader, "boxMax");
            p.locPiece = GetShaderLocation(p.shader, "pieceNumber");
        }
        return &(m_Programs[key] = p);
    }

    bool GpuSilhouettes::Draw(const TraceRequest& rq, TraceResult* out) {
        const double start = GetTime();
        const int nv = (int)rq.views.size(), res = rq.res;
        if (!m_Ok || nv < 1 || nv > 24 || res < 1 || (long long)nv * res > 16384) return false;
        // every piece's shader first: one that won't build (too many uniforms for this driver, say) means the CPU draws
        std::vector<Program*> programs;
        for (const TracePiece& piece : rq.pieces) {
            Program* p = piece.params.empty() ? nullptr : ProgramFor(piece.code, (int)piece.params.size());
            if (p && !p->ok) return false;
            programs.push_back(p);
        }
        const int width = nv * res, height = res;
        if (!Target(width, height)) return false;
        std::vector<Vector3> dir(nv), u(nv), v(nv);
        for (int k = 0; k < nv; k++) { dir[k] = rq.views[k].dir; u[k] = rq.views[k].u; v[k] = rq.views[k].v; }
        const float fres = (float)res;

        BeginTextureMode(m_Target);
        rlClearColor(0, 0, 0, 0);
        rlClearScreenBuffers();
        rlDisableColorBlend();
        rlEnableDepthTest();
        rlEnableDepthMask();
        // the last drawn wins ties (pixels two pieces cover fully): the first piece last, as on the CPU
        for (int i = (int)rq.pieces.size() - 1; i >= 0; i--) {
            const TracePiece& piece = rq.pieces[i];
            Program* p = programs[i];
            if (!p) continue;
            const float farAway = 4.0f * rq.half + Vector3Distance(piece.box.min, piece.box.max), number = (float)(i + 1);
            BeginShaderMode(p->shader);
            SetShaderValueV(p->shader, p->locU, piece.params.data(), SHADER_UNIFORM_VEC4, (int)piece.params.size());
            SetShaderValueV(p->shader, p->locDir, dir.data(), SHADER_UNIFORM_VEC3, nv);
            SetShaderValueV(p->shader, p->locViewU, u.data(), SHADER_UNIFORM_VEC3, nv);
            SetShaderValueV(p->shader, p->locViewV, v.data(), SHADER_UNIFORM_VEC3, nv);
            SetShaderValue(p->shader, p->locCentre, &rq.centre, SHADER_UNIFORM_VEC3);
            SetShaderValue(p->shader, p->locHalf, &rq.half, SHADER_UNIFORM_FLOAT);
            SetShaderValue(p->shader, p->locRes, &fres, SHADER_UNIFORM_FLOAT);
            SetShaderValue(p->shader, p->locTau, &rq.tau, SHADER_UNIFORM_FLOAT);
            SetShaderValue(p->shader, p->locFar, &farAway, SHADER_UNIFORM_FLOAT);
            SetShaderValue(p->shader, p->locQuat, &piece.xf.rotation, SHADER_UNIFORM_VEC4);
            SetShaderValue(p->shader, p->locPos, &piece.xf.position, SHADER_UNIFORM_VEC3);
            SetShaderValue(p->shader, p->locMin, &piece.box.min, SHADER_UNIFORM_VEC3);
            SetShaderValue(p->shader, p->locMax, &piece.box.max, SHADER_UNIFORM_VEC3);
            SetShaderValue(p->shader, p->locPiece, &number, SHADER_UNIFORM_FLOAT);
            DrawRectangle(0, 0, width, height, WHITE);
            EndShaderMode();
        }
        rlDrawRenderBatchActive();
        rlDisableDepthTest();
        rlEnableColorBlend();
        EndTextureMode();

        float* px = (float*)rlReadTexturePixels(m_Target.texture.id, width, height, RL_PIXELFORMAT_UNCOMPRESSED_R32G32B32A32);
        if (!px) return false;
        const size_t count = (size_t)nv * res * res;
        out->nearest.assign(count, 0.0f);
        out->along.assign(count, 0.0f);
        out->piece.assign(count, -1);
        for (int y = 0; y < res; y++)
            for (int x = 0; x < width; x++) {
                const float* t = px + ((size_t)y * width + x) * 4;
                size_t k = (size_t)(x / res) * res * res + (size_t)y * res + (x % res);
                out->nearest[k] = t[0];
                out->along[k] = t[1];
                out->piece[k] = (int)lroundf(t[2]) - 1;
            }
        MemFree(px);
        m_Drawn++;
        m_LastMs = (GetTime() - start) * 1000.0;
        return true;
    }
}
