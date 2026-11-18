#pragma once
// The generator's silhouettes drawn on the GPU (autodesign/tracer.hpp), with the editor's OpenGL context.
//
// The context belongs to the UI thread, and the search runs on another. So the search's thread only asks: Trace hands
// its request over and waits. The UI thread answers in Service, called once a frame: it draws what was asked, reads it
// back, and while a search keeps asking it stays there for most of the frame (the editor then redraws a dozen times a
// second rather than sixty: it stalls between the search's steps, which is the price of sharing the context).
//
// Drawing: one float target holds every view side by side. Each piece is one rectangle over all of it, with a shader
// made from the piece's field (cached by its code: a piece's numbers are uniforms, so only a rewrite compiles
// anything), marching each pixel's ray as Loss::Trace does. The depth test keeps, per pixel, the piece covering it most.
// For the semantic pictures (everyPiece) each piece has a band of its own instead, one view-height tall, below the last.
#include "autodesign/tracer.hpp"
#include <raylib.h>
#include <condition_variable>
#include <mutex>
#include <string>
#include <unordered_map>

namespace Bakugan {
    class GpuSilhouettes : public SilhouetteTracer {
    public:
        // the search's thread
        bool Trace(const TraceRequest& request, TraceResult* result) override;
        std::string Name() const override { return "the GPU"; }

        // the UI thread (the one with the context)
        bool Init();                         // false: no float targets here (Error says)
        void Service(double seconds);        // answers requests; with seconds > 0, waits that long for more when one came lately
        void Close();                        // no more answers (before the search's thread is joined, or the context goes)
        void Unload();
        bool Ok() const { return m_Ok; }
        const std::string& Error() const { return m_Error; }
        // for the UI: how many drawings so far, how long the last took (ms), and shaders compiled
        int Drawn() const { return m_Drawn; }
        double LastMs() const { return m_LastMs; }
        int Compiles() const { return m_Compiles; }
        // Draws a request right here (the UI thread): what Service does for the search.
        bool Draw(const TraceRequest& request, TraceResult* result);
    private:
        struct Program {
            Shader shader = {};
            bool ok = false;
            int locU = -1, locDir = -1, locViewU = -1, locViewV = -1, locCentre = -1, locHalf = -1, locRes = -1, locTau = -1, locFar = -1,
                locQuat = -1, locPos = -1, locMin = -1, locMax = -1, locPiece = -1, locEvery = -1, locRow = -1;
        };
        Program* ProgramFor(const std::string& code, int uniforms);
        bool Target(int width, int height);
        void Drop();
    private:
        bool m_Ok = false;
        std::string m_Error;
        RenderTexture2D m_Target = {};
        std::unordered_map<std::string, Program> m_Programs;
        int m_Drawn = 0, m_Compiles = 0;
        double m_LastMs = 0.0;
        // the hand-off
        std::mutex m_Mutex;
        std::condition_variable m_Asked, m_Answered;
        const TraceRequest* m_Request = nullptr;
        TraceResult* m_Result = nullptr;
        bool m_Pending = false, m_Done = false, m_Success = false, m_Closed = false;
        double m_LastAsked = -1e9;           // GetTime() of the last request
    };
}
