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

    bool GpuSilhouettes::Evaluate(const EvalRequest& request, std::vector<double>* sums) {
        std::unique_lock<std::mutex> l(m_Mutex);
        if (m_Closed || !m_Ok || m_Pending) return false;
        m_EvalRequest = &request;
        m_EvalSums = sums;
        m_Pending = true;
        m_Done = false;
        m_Asked.notify_all();
        m_Answered.wait_for(l, std::chrono::seconds(4), [&] { return m_Done || m_Closed; });
        bool ok = m_Done && m_Success;
        m_Pending = false;
        m_EvalRequest = nullptr;
        m_EvalSums = nullptr;
        return ok;
    }

    void GpuSilhouettes::Service(double seconds) {
        using Clock = std::chrono::steady_clock;
        const Clock::time_point until = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(std::max(0.0, seconds)));
        std::unique_lock<std::mutex> l(m_Mutex);
        for (;;) {
            if (m_Pending && !m_Done) {
                m_Success = m_EvalRequest ? DrawEval(*m_EvalRequest, m_EvalSums) : Draw(*m_Request, m_Result);
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
        if (m_EvalFbo) rlUnloadFramebuffer(m_EvalFbo);   // (and its texture)
        m_EvalFbo = m_EvalTexture = 0;
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
            p.locEvery = GetShaderLocation(p.shader, "everyPiece");
            p.locRow = GetShaderLocation(p.shader, "pieceRow");
        }
        return &(m_Programs[key] = p);
    }

    // More views than the shader's arrays hold (24) are drawn 24 at a time, and the results put back in order.
    bool GpuSilhouettes::Draw(const TraceRequest& rq, TraceResult* out) {
        const int nv = (int)rq.views.size();
        if (nv <= 24) return DrawViews(rq, out);
        const double start = GetTime();
        const size_t px = (size_t)rq.res * rq.res;
        const int bands = rq.everyPiece ? (int)rq.pieces.size() : 1;
        const size_t count = (size_t)bands * nv * px;
        out->nearest.assign(count, 0.0f);
        out->along.assign(count, 0.0f);
        out->piece.assign(count, -1);
        out->enter.assign(rq.everyPiece ? count : 0, -1.0f);
        for (int v0 = 0; v0 < nv; v0 += 24) {
            const int n = std::min(24, nv - v0);
            TraceRequest part = rq;
            part.views.assign(rq.views.begin() + v0, rq.views.begin() + v0 + n);
            TraceResult r;
            if (!DrawViews(part, &r)) return false;
            m_Drawn--;                                   // (one drawing, however many parts)
            for (int band = 0; band < bands; band++)
                for (int v = 0; v < n; v++) {
                    const size_t from = ((size_t)band * n + v) * px, to = ((size_t)band * nv + v0 + v) * px;
                    std::copy_n(r.nearest.begin() + from, px, out->nearest.begin() + to);
                    std::copy_n(r.along.begin() + from, px, out->along.begin() + to);
                    std::copy_n(r.piece.begin() + from, px, out->piece.begin() + to);
                    if (rq.everyPiece) std::copy_n(r.enter.begin() + from, px, out->enter.begin() + to);
                }
        }
        m_Drawn++;
        m_LastMs = (GetTime() - start) * 1000.0;
        return true;
    }

    bool GpuSilhouettes::DrawViews(const TraceRequest& rq, TraceResult* out) {
        const double start = GetTime();
        const int nv = (int)rq.views.size(), res = rq.res;
        const int bands = rq.everyPiece ? (int)rq.pieces.size() : 1;
        if (!m_Ok || nv < 1 || nv > 24 || res < 1 || bands < 1 || (long long)nv * res > 16384 || (long long)bands * res > 16384) return false;
        // every piece's shader first: one that won't build (too many uniforms for this driver, say) means the CPU draws
        std::vector<Program*> programs;
        for (const TracePiece& piece : rq.pieces) {
            Program* p = piece.params.empty() ? nullptr : ProgramFor(piece.code, (int)piece.params.size());
            if (p && !p->ok) return false;
            programs.push_back(p);
        }
        const int width = nv * res, height = res * bands;
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
            const float every = rq.everyPiece ? 1.0f : 0.0f, row = rq.everyPiece ? (float)i : 0.0f;
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
            SetShaderValue(p->shader, p->locEvery, &every, SHADER_UNIFORM_FLOAT);
            SetShaderValue(p->shader, p->locRow, &row, SHADER_UNIFORM_FLOAT);
            DrawRectangle(0, 0, width, height, WHITE);
            EndShaderMode();
        }
        rlDrawRenderBatchActive();
        rlDisableDepthTest();
        rlEnableColorBlend();
        EndTextureMode();

        float* px = (float*)rlReadTexturePixels(m_Target.texture.id, width, height, RL_PIXELFORMAT_UNCOMPRESSED_R32G32B32A32);
        if (!px) return false;
        const size_t count = (size_t)bands * nv * res * res;
        out->nearest.assign(count, 0.0f);
        out->along.assign(count, 0.0f);
        out->piece.assign(count, -1);
        out->enter.assign(rq.everyPiece ? count : 0, -1.0f);
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++) {
                const float* t = px + ((size_t)y * width + x) * 4;
                const int band = y / res, row = y % res;
                size_t k = ((size_t)band * nv + (size_t)(x / res)) * res * res + (size_t)row * res + (x % res);
                if (rq.everyPiece) {
                    if (t[3] < 0.5f) continue;                  // (cleared: the piece doesn't cover it)
                    out->nearest[k] = t[0];
                    out->along[k] = t[1];
                    out->enter[k] = t[2];
                    out->piece[k] = band;
                    continue;
                }
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

namespace Bakugan {
    GpuSilhouettes::Program* GpuSilhouettes::EvalProgramFor(const std::string& code, int uniforms) {
        std::string key = "eval\n" + std::to_string(uniforms) + "\n" + code;
        auto it = m_Programs.find(key);
        if (it != m_Programs.end()) return &it->second;
        if (m_Programs.size() >= 256) {
            for (auto& kv : m_Programs)
                if (kv.second.ok) UnloadShader(kv.second.shader);
            m_Programs.clear();
        }
        Program p;
        std::string fs = Sdf::EvalFragmentSource(code, uniforms);
        p.shader = LoadShaderFromMemory(Sdf::TraceVertexSource(), fs.c_str());
        m_Compiles++;
        p.ok = p.shader.id != 0 && p.shader.id != rlGetShaderIdDefault();
        if (p.ok) {
            p.locU = GetShaderLocation(p.shader, "U");
            p.locPoints = GetShaderLocation(p.shader, "points");
            p.locQuat = GetShaderLocation(p.shader, "bodyQuat");
            p.locPos = GetShaderLocation(p.shader, "bodyPos");
            p.locStart = GetShaderLocation(p.shader, "start");
            p.locCount = GetShaderLocation(p.shader, "count");
            p.locChunkStart = GetShaderLocation(p.shader, "chunkStart");
            p.locOutWidth = GetShaderLocation(p.shader, "outWidth");
            p.locPointsWidth = GetShaderLocation(p.shader, "pointsWidth");
        }
        return &m_Programs.emplace(key, p).first->second;
    }

    // Every job's points summed on the GPU, Sdf::EvalChunk to a pixel; read back once and summed here per job.
    bool GpuSilhouettes::DrawEval(const EvalRequest& rq, std::vector<double>* sums) {
        const double start = GetTime();
        sums->assign(rq.jobs.size(), 0.0);
        if (!m_Ok) return false;
        if (rq.jobs.empty()) return true;
        const int chunk = Sdf::EvalChunk;
        std::vector<long long> first(rq.jobs.size());
        long long chunks = 0;
        for (size_t j = 0; j < rq.jobs.size(); j++) {
            first[j] = chunks;
            chunks += (std::max(0, rq.jobs[j].count) + chunk - 1) / chunk;
        }
        if (chunks == 0) return true;
        const int outWidth = 2048, pointsWidth = 4096;
        const long long outHeight = (chunks + outWidth - 1) / outWidth;
        const long long pointsHeight = ((long long)rq.points.size() + pointsWidth - 1) / pointsWidth;
        if (outHeight > 16384 || pointsHeight > 16384 || pointsHeight < 1) return false;

        // the target: a float texture, kept while big enough
        if (!m_EvalFbo || m_EvalWidth != outWidth || m_EvalHeight < outHeight) {
            if (m_EvalFbo) rlUnloadFramebuffer(m_EvalFbo);
            m_EvalWidth = outWidth;
            m_EvalHeight = (int)std::max<long long>(outHeight, 64);
            m_EvalFbo = rlLoadFramebuffer();
            if (!m_EvalFbo) return false;
            rlEnableFramebuffer(m_EvalFbo);
            m_EvalTexture = rlLoadTexture(nullptr, m_EvalWidth, m_EvalHeight, RL_PIXELFORMAT_UNCOMPRESSED_R32G32B32A32, 1);
            rlFramebufferAttach(m_EvalFbo, m_EvalTexture, RL_ATTACHMENT_COLOR_CHANNEL0, RL_ATTACHMENT_TEXTURE2D, 0);
            const bool complete = m_EvalTexture != 0 && rlFramebufferComplete(m_EvalFbo);
            rlDisableFramebuffer();
            if (!complete) {
                rlUnloadFramebuffer(m_EvalFbo);
                m_EvalFbo = m_EvalTexture = 0;
                return false;
            }
        }
        // the points: a float texture too
        std::vector<Vector4> pts((size_t)pointsWidth * pointsHeight, Vector4{ 0, 0, 0, 0 });
        std::copy(rq.points.begin(), rq.points.end(), pts.begin());
        Texture2D points = {};
        points.id = rlLoadTexture(pts.data(), pointsWidth, (int)pointsHeight, RL_PIXELFORMAT_UNCOMPRESSED_R32G32B32A32, 1);
        if (points.id == 0) return false;
        points.width = pointsWidth;
        points.height = (int)pointsHeight;
        points.mipmaps = 1;
        points.format = PIXELFORMAT_UNCOMPRESSED_R32G32B32A32;
        rlTextureParameters(points.id, RL_TEXTURE_MIN_FILTER, RL_TEXTURE_FILTER_NEAREST);
        rlTextureParameters(points.id, RL_TEXTURE_MAG_FILTER, RL_TEXTURE_FILTER_NEAREST);

        RenderTexture2D target = {};
        target.id = m_EvalFbo;
        target.texture.id = m_EvalTexture;
        target.texture.width = m_EvalWidth;
        target.texture.height = m_EvalHeight;
        target.texture.mipmaps = 1;
        target.texture.format = PIXELFORMAT_UNCOMPRESSED_R32G32B32A32;
        bool ok = true;
        BeginTextureMode(target);
        rlClearColor(0, 0, 0, 0);
        rlClearScreenBuffers();
        rlDisableColorBlend();
        const float fOut = (float)outWidth, fPoints = (float)pointsWidth;
        for (size_t j = 0; j < rq.jobs.size(); j++) {
            const EvalJob& job = rq.jobs[j];
            if (job.count <= 0) continue;
            Program* p = EvalProgramFor(rq.codes[job.code], (int)std::max<size_t>(1, job.params.size()));
            if (!p || !p->ok) { ok = false; break; }
            // the rows its chunks are on (gl_FragCoord rows: raylib's texture mode counts from the top)
            const long long c0 = first[j], c1 = first[j] + (job.count + chunk - 1) / chunk - 1;
            const int r0 = (int)(c0 / outWidth), r1 = (int)(c1 / outWidth);
            const float fStart = (float)job.start, fCount = (float)job.count, fChunk = (float)c0;
            BeginShaderMode(p->shader);
            if (!job.params.empty()) SetShaderValueV(p->shader, p->locU, job.params.data(), SHADER_UNIFORM_VEC4, (int)job.params.size());
            SetShaderValueTexture(p->shader, p->locPoints, points);
            SetShaderValue(p->shader, p->locQuat, &job.xf.rotation, SHADER_UNIFORM_VEC4);
            SetShaderValue(p->shader, p->locPos, &job.xf.position, SHADER_UNIFORM_VEC3);
            SetShaderValue(p->shader, p->locStart, &fStart, SHADER_UNIFORM_FLOAT);
            SetShaderValue(p->shader, p->locCount, &fCount, SHADER_UNIFORM_FLOAT);
            SetShaderValue(p->shader, p->locChunkStart, &fChunk, SHADER_UNIFORM_FLOAT);
            SetShaderValue(p->shader, p->locOutWidth, &fOut, SHADER_UNIFORM_FLOAT);
            SetShaderValue(p->shader, p->locPointsWidth, &fPoints, SHADER_UNIFORM_FLOAT);
            DrawRectangle(0, m_EvalHeight - 1 - r1, outWidth, r1 - r0 + 1, WHITE);
            EndShaderMode();
        }
        rlDrawRenderBatchActive();
        rlEnableColorBlend();
        EndTextureMode();
        rlUnloadTexture(points.id);
        if (!ok) return false;

        float* px = (float*)rlReadTexturePixels(m_EvalTexture, m_EvalWidth, m_EvalHeight, RL_PIXELFORMAT_UNCOMPRESSED_R32G32B32A32);
        if (!px) return false;
        for (size_t j = 0; j < rq.jobs.size(); j++) {
            const long long n = (std::max(0, rq.jobs[j].count) + chunk - 1) / chunk;
            double s = 0.0;
            for (long long k = 0; k < n; k++) s += px[(size_t)(first[j] + k) * 4];
            (*sums)[j] = s;
        }
        MemFree(px);
        m_Evaluated++;
        m_LastEvalMs = (GetTime() - start) * 1000.0;
        return true;
    }
}
