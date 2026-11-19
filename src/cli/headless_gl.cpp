#include "headless_gl.hpp"
#include <rlgl.h>
#include <cstdint>
#include <cstring>
#include <vector>

#ifdef __linux__
#include <dlfcn.h>
#endif

namespace Bakugan::Cli {
#ifndef __linux__
    bool HeadlessGl::Open(int, std::string* why) {
        *why = "a headless OpenGL context is made only on Linux";
        return false;
    }
    void HeadlessGl::Close() {}
#else
    namespace {
        // EGL, the little of it needed (egl.h / eglext.h values), so the build needs no EGL headers either
        using EGLint = int32_t;
        using EGLBoolean = unsigned int;
        using EGLenum = unsigned int;
        using EGLDisplay = void*;
        using EGLConfig = void*;
        using EGLContext = void*;
        using EGLSurface = void*;
        using EGLDeviceEXT = void*;
        constexpr EGLint NONE = 0x3038, SURFACE_TYPE = 0x3033, PBUFFER_BIT = 0x0001, RENDERABLE_TYPE = 0x3040, OPENGL_BIT = 0x0008,
                         WIDTH = 0x3057, HEIGHT = 0x3056, EXTENSIONS = 0x3055, CONTEXT_MAJOR = 0x3098, CONTEXT_MINOR = 0x30FB,
                         PROFILE_MASK = 0x30FD, CORE_PROFILE = 0x1, DRM_DEVICE_FILE = 0x3233;
        constexpr EGLenum OPENGL_API = 0x30A2, PLATFORM_DEVICE = 0x313F, PLATFORM_SURFACELESS = 0x31DD;
        constexpr unsigned int GL_RENDERER = 0x1F01;

        struct Egl {
            void* (*getProcAddress)(const char*) = nullptr;
            EGLBoolean (*initialize)(EGLDisplay, EGLint*, EGLint*) = nullptr;
            EGLBoolean (*terminate)(EGLDisplay) = nullptr;
            EGLBoolean (*bindApi)(EGLenum) = nullptr;
            EGLBoolean (*chooseConfig)(EGLDisplay, const EGLint*, EGLConfig*, EGLint, EGLint*) = nullptr;
            EGLContext (*createContext)(EGLDisplay, EGLConfig, EGLContext, const EGLint*) = nullptr;
            EGLBoolean (*destroyContext)(EGLDisplay, EGLContext) = nullptr;
            EGLSurface (*createPbuffer)(EGLDisplay, EGLConfig, const EGLint*) = nullptr;
            EGLBoolean (*destroySurface)(EGLDisplay, EGLSurface) = nullptr;
            EGLBoolean (*makeCurrent)(EGLDisplay, EGLSurface, EGLSurface, EGLContext) = nullptr;
            const char* (*queryString)(EGLDisplay, EGLint) = nullptr;
            EGLint (*getError)() = nullptr;
            // extensions
            EGLBoolean (*queryDevices)(EGLint, EGLDeviceEXT*, EGLint*) = nullptr;
            const char* (*queryDeviceString)(EGLDeviceEXT, EGLint) = nullptr;
            EGLDisplay (*getPlatformDisplay)(EGLenum, void*, const EGLint*) = nullptr;
        };
        Egl g;

        template <class F> bool Load(void* lib, const char* name, F& f) {
            f = (F)dlsym(lib, name);
            return f != nullptr;
        }

        bool HasExtension(const char* list, const char* name) {
            if (!list) return false;
            const size_t n = strlen(name);
            for (const char* p = list; (p = strstr(p, name)) != nullptr; p += n)
                if ((p == list || p[-1] == ' ') && (p[n] == ' ' || p[n] == '\0')) return true;
            return false;
        }

        bool Software(const std::string& renderer) {
            return renderer.find("llvmpipe") != std::string::npos || renderer.find("softpipe") != std::string::npos ||
                   renderer.find("Software") != std::string::npos || renderer.find("swrast") != std::string::npos;
        }
    }

    bool HeadlessGl::Open(int device, std::string* why) {
        Close();
        m_Lib = dlopen("libEGL.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!m_Lib) m_Lib = dlopen("libEGL.so", RTLD_NOW | RTLD_LOCAL);
        if (!m_Lib) {
            *why = "no libEGL (install your GPU's EGL: on Arch it comes with the driver, libglvnd + nvidia-utils or mesa)";
            return false;
        }
        bool ok = Load(m_Lib, "eglGetProcAddress", g.getProcAddress) && Load(m_Lib, "eglInitialize", g.initialize) &&
                  Load(m_Lib, "eglTerminate", g.terminate) && Load(m_Lib, "eglBindAPI", g.bindApi) && Load(m_Lib, "eglChooseConfig", g.chooseConfig) &&
                  Load(m_Lib, "eglCreateContext", g.createContext) && Load(m_Lib, "eglDestroyContext", g.destroyContext) &&
                  Load(m_Lib, "eglCreatePbufferSurface", g.createPbuffer) && Load(m_Lib, "eglDestroySurface", g.destroySurface) &&
                  Load(m_Lib, "eglMakeCurrent", g.makeCurrent) && Load(m_Lib, "eglQueryString", g.queryString) && Load(m_Lib, "eglGetError", g.getError);
        if (!ok) {
            *why = "libEGL is missing functions";
            Close();
            return false;
        }
        g.queryDevices = (decltype(g.queryDevices))g.getProcAddress("eglQueryDevicesEXT");
        g.queryDeviceString = (decltype(g.queryDeviceString))g.getProcAddress("eglQueryDeviceStringEXT");
        g.getPlatformDisplay = (decltype(g.getPlatformDisplay))g.getProcAddress("eglGetPlatformDisplayEXT");
        const char* client = g.queryString(nullptr, EXTENSIONS);
        if (!g.getPlatformDisplay) {
            *why = "this EGL can't open a display without a window system (no eglGetPlatformDisplayEXT)";
            Close();
            return false;
        }

        // the candidates: every EGL device (the GPUs), then Mesa's surfaceless platform
        struct Candidate {
            EGLenum platform;
            void* native;
            std::string file;
        };
        std::vector<Candidate> candidates;
        if (g.queryDevices && HasExtension(client, "EGL_EXT_platform_device")) {
            EGLDeviceEXT devices[16];
            EGLint n = 0;
            if (g.queryDevices(16, devices, &n))
                for (EGLint k = 0; k < n; k++) {
                    const char* file = g.queryDeviceString ? g.queryDeviceString(devices[k], DRM_DEVICE_FILE) : nullptr;
                    candidates.push_back({ PLATFORM_DEVICE, devices[k], file ? file : "" });
                }
        }
        if (device >= (int)candidates.size()) {
            *why = "there is no EGL device " + std::to_string(device) + " (" + std::to_string(candidates.size()) + " found)";
            Close();
            return false;
        }
        if (device >= 0) candidates = { candidates[device] };
        else if (HasExtension(client, "EGL_MESA_platform_surfaceless")) candidates.push_back({ PLATFORM_SURFACELESS, nullptr, "surfaceless" });
        if (candidates.empty()) {
            *why = "this EGL offers no devices and no surfaceless platform";
            Close();
            return false;
        }

        // the first hardware renderer that gives a 3.3 core context (a software one only if there's nothing else)
        std::string lastWhy = "no EGL display gave an OpenGL 3.3 context";
        for (int pass = 0; pass < 2 && !m_Context; pass++)
            for (const Candidate& cand : candidates) {
                EGLDisplay dpy = g.getPlatformDisplay(cand.platform, cand.native, nullptr);
                EGLint major = 0, minor = 0;
                if (!dpy || !g.initialize(dpy, &major, &minor)) continue;
                const EGLint configAttribs[] = { SURFACE_TYPE, PBUFFER_BIT, RENDERABLE_TYPE, OPENGL_BIT, NONE };
                EGLConfig config = nullptr;
                EGLint count = 0;
                if (!g.bindApi(OPENGL_API) || !g.chooseConfig(dpy, configAttribs, &config, 1, &count) || count < 1) {
                    // (surfaceless displays may offer no pbuffer configs: any OpenGL one will do)
                    const EGLint anyAttribs[] = { RENDERABLE_TYPE, OPENGL_BIT, NONE };
                    if (!g.chooseConfig(dpy, anyAttribs, &config, 1, &count) || count < 1) {
                        g.terminate(dpy);
                        lastWhy = "no OpenGL config";
                        continue;
                    }
                }
                const EGLint contextAttribs[] = { CONTEXT_MAJOR, 3, CONTEXT_MINOR, 3, PROFILE_MASK, CORE_PROFILE, NONE };
                EGLContext ctx = g.createContext(dpy, config, nullptr, contextAttribs);
                if (!ctx) {
                    g.terminate(dpy);
                    lastWhy = "no OpenGL 3.3 core context";
                    continue;
                }
                EGLSurface surface = nullptr;
                bool current = HasExtension(g.queryString(dpy, EXTENSIONS), "EGL_KHR_surfaceless_context") && g.makeCurrent(dpy, nullptr, nullptr, ctx);
                if (!current) {
                    const EGLint pbufferAttribs[] = { WIDTH, 16, HEIGHT, 16, NONE };
                    surface = g.createPbuffer(dpy, config, pbufferAttribs);
                    current = surface && g.makeCurrent(dpy, surface, surface, ctx);
                }
                if (!current) {
                    if (surface) g.destroySurface(dpy, surface);
                    g.destroyContext(dpy, ctx);
                    g.terminate(dpy);
                    lastWhy = "couldn't make the context current";
                    continue;
                }
                auto getString = (const unsigned char* (*)(unsigned int))g.getProcAddress("glGetString");
                const unsigned char* r = getString ? getString(GL_RENDERER) : nullptr;
                std::string renderer = r ? (const char*)r : "unknown";
                if (pass == 0 && device < 0 && Software(renderer)) {       // look for a real GPU first
                    g.makeCurrent(dpy, nullptr, nullptr, nullptr);
                    if (surface) g.destroySurface(dpy, surface);
                    g.destroyContext(dpy, ctx);
                    g.terminate(dpy);
                    continue;
                }
                m_Display = dpy;
                m_Context = ctx;
                m_Surface = surface;
                m_Renderer = renderer + (cand.file.empty() ? "" : " (" + cand.file + ")");
                break;
            }
        if (!m_Context) {
            *why = lastWhy;
            Close();
            return false;
        }
        // raylib's OpenGL layer on it: the functions loaded through EGL, then its state (default shader, batch, ...)
        rlLoadExtensions((void*)g.getProcAddress);
        rlglInit(64, 64);
        return true;
    }

    void HeadlessGl::Close() {
        if (m_Context) {
            rlglClose();
            g.makeCurrent(m_Display, nullptr, nullptr, nullptr);
            if (m_Surface) g.destroySurface(m_Display, m_Surface);
            g.destroyContext(m_Display, m_Context);
            g.terminate(m_Display);
        }
        m_Context = m_Surface = m_Display = nullptr;
        if (m_Lib) dlclose(m_Lib);
        m_Lib = nullptr;
    }
#endif
}
