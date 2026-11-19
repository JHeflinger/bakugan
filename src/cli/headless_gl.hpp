#pragma once
// An OpenGL 3.3 core context with no window and no display (EGL: NVIDIA's device platform, Mesa's surfaceless one), so
// the command line can use the GPU from tmux, over ssh, or on a server. libEGL is loaded when asked for (dlopen): the
// build doesn't need it, and where it isn't there Open says so. Linux only; elsewhere Open is false.
#include <string>

namespace Bakugan::Cli {
    class HeadlessGl {
    public:
        // device: which EGL device (-1: the first that isn't a software renderer, else any). On success the context is
        // current on this thread and rlgl is set up on it (raylib's drawing functions work without a window).
        bool Open(int device, std::string* why);
        void Close();
        ~HeadlessGl() { Close(); }
        bool IsOpen() const { return m_Context != nullptr; }
        const std::string& Renderer() const { return m_Renderer; }   // GL_RENDERER, and the device file when known
    private:
        void* m_Lib = nullptr;
        void* m_Display = nullptr;
        void* m_Context = nullptr;
        void* m_Surface = nullptr;          // (a 16 x 16 pbuffer, only where surfaceless contexts aren't offered)
        std::string m_Renderer;
    };
}
