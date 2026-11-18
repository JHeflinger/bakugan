#include "dino.hpp"
#include "vendor/onnxruntime/onnxruntime_c_api.h"
#include <cstring>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace Bakugan {
    namespace {
        // The oldest ONNX Runtime API this needs (1.16): newer runtimes all serve it.
        constexpr uint32_t API = 16;

#if defined(_WIN32)
        const char* DefaultLibrary = "onnxruntime.dll";
#elif defined(__APPLE__)
        const char* DefaultLibrary = "libonnxruntime.dylib";
#else
        const char* DefaultLibrary = "libonnxruntime.so";
#endif

        void* OpenLibrary(const std::string& path, std::string* err) {
#ifdef _WIN32
            HMODULE h = LoadLibraryA(path.c_str());
            if (!h) *err = "can't load " + path + " (error " + std::to_string(GetLastError()) + ")";
            return (void*)h;
#else
            void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!h) {
                const char* e = dlerror();
                *err = e ? e : "can't load " + path;
            }
            return h;
#endif
        }

        void* Symbol(void* lib, const char* name) {
#ifdef _WIN32
            return (void*)GetProcAddress((HMODULE)lib, name);
#else
            return dlsym(lib, name);
#endif
        }

        void CloseLibrary(void* lib) {
#ifdef _WIN32
            FreeLibrary((HMODULE)lib);
#else
            dlclose(lib);
#endif
        }

        std::basic_string<ORTCHAR_T> OrtPath(const std::string& utf8) {
#ifdef _WIN32
            int n = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, nullptr, 0);
            std::wstring w(n > 0 ? n - 1 : 0, L'\0');
            if (n > 1) MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, w.data(), n);
            return w;
#else
            return utf8;
#endif
        }
    }

    struct Dino::Runtime {
        void* lib = nullptr;
        const OrtApi* api = nullptr;
        OrtEnv* env = nullptr;
        OrtSessionOptions* options = nullptr;
        OrtSession* session = nullptr;
        OrtMemoryInfo* memory = nullptr;

        // An ORT call's failure as text (and the status freed); empty when it worked.
        std::string Check(OrtStatus* s) {
            if (!s) return "";
            std::string e = api->GetErrorMessage(s);
            api->ReleaseStatus(s);
            return e.empty() ? "ONNX Runtime error" : e;
        }

        ~Runtime() {
            if (api) {
                if (memory) api->ReleaseMemoryInfo(memory);
                if (session) api->ReleaseSession(session);
                if (options) api->ReleaseSessionOptions(options);
                if (env) api->ReleaseEnv(env);
            }
            if (lib) CloseLibrary(lib);
        }
    };

    Dino::Dino(const std::string& library, const std::string& model, bool gpu) : m_R(std::make_unique<Runtime>()) {
        Runtime& r = *m_R;
        std::string lib = library.empty() ? DefaultLibrary : library;
        if (model.empty()) { m_Error = "no DINOv3 model chosen (make one with tools/autodesign/export_dinov3.py)"; return; }
        r.lib = OpenLibrary(lib, &m_Error);
        if (!r.lib) { m_Error = "ONNX Runtime: " + m_Error; return; }
        using GetBase = const OrtApiBase*(ORT_API_CALL*)();
        auto getBase = (GetBase)Symbol(r.lib, "OrtGetApiBase");
        if (!getBase) { m_Error = lib + " isn't ONNX Runtime (no OrtGetApiBase)"; return; }
        r.api = getBase()->GetApi(API);
        if (!r.api) { m_Error = std::string("ONNX Runtime ") + getBase()->GetVersionString() + " is too old: 1.16 or newer is needed"; return; }

        const OrtApi& a = *r.api;
        if (!(m_Error = r.Check(a.CreateEnv(ORT_LOGGING_LEVEL_WARNING, "bakugan", &r.env))).empty()) return;
        if (!(m_Error = r.Check(a.CreateSessionOptions(&r.options))).empty()) return;
        r.Check(a.SetSessionGraphOptimizationLevel(r.options, ORT_ENABLE_ALL));
        if (gpu) {
            OrtCUDAProviderOptionsV2* cuda = nullptr;
            std::string e = r.Check(a.CreateCUDAProviderOptions(&cuda));
            if (e.empty()) e = r.Check(a.SessionOptionsAppendExecutionProvider_CUDA_V2(r.options, cuda));
            if (cuda) a.ReleaseCUDAProviderOptions(cuda);
            m_Device = e.empty() ? "CUDA" : "CPU (no CUDA: " + e + ")";
        }
        std::basic_string<ORTCHAR_T> path = OrtPath(model);
        if (!(m_Error = r.Check(a.CreateSession(r.env, path.c_str(), r.options, &r.session))).empty()) {
            m_Error = "can't load " + model + ": " + m_Error;
            return;
        }
        if (!(m_Error = r.Check(a.CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &r.memory))).empty()) return;

        // the size it was exported for: render is [1, 1, S, S]
        size_t inputs = 0, outputs = 0;
        r.Check(a.SessionGetInputCount(r.session, &inputs));
        r.Check(a.SessionGetOutputCount(r.session, &outputs));
        if (inputs != 3 || outputs != 3) { m_Error = model + " isn't a DINOv3 loss graph (see export_dinov3.py)"; return; }
        OrtTypeInfo* info = nullptr;
        if (!(m_Error = r.Check(a.SessionGetInputTypeInfo(r.session, 0, &info))).empty()) return;
        const OrtTensorTypeAndShapeInfo* tensor = nullptr;
        size_t dims = 0;
        int64_t shape[4] = { 0, 0, 0, 0 };
        if (r.Check(a.CastTypeInfoToTensorInfo(info, &tensor)).empty() && tensor && r.Check(a.GetDimensionsCount(tensor, &dims)).empty() && dims == 4)
            r.Check(a.GetDimensions(tensor, shape, 4));
        a.ReleaseTypeInfo(info);
        if (dims != 4 || shape[0] != 1 || shape[1] != 1 || shape[2] <= 0 || shape[2] != shape[3]) {
            m_Error = model + " isn't a DINOv3 loss graph for square views (see export_dinov3.py)";
            return;
        }
        m_Size = (int)shape[2];
    }

    Dino::~Dino() = default;

    bool Dino::Run(const float* render, const float* target, float patchWeight, float* loss, float* gradRender, float* gradTarget) {
        if (!Ok()) return false;
        Runtime& r = *m_R;
        const OrtApi& a = *r.api;
        const int64_t shape[4] = { 1, 1, m_Size, m_Size }, one[1] = { 1 };
        const size_t px = (size_t)m_Size * m_Size;
        OrtValue* in[3] = { nullptr, nullptr, nullptr };
        OrtValue* out[3] = { nullptr, nullptr, nullptr };
        std::string e;
        // ORT doesn't write to inputs; its API just isn't const
        e = r.Check(a.CreateTensorWithDataAsOrtValue(r.memory, (void*)render, px * sizeof(float), shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[0]));
        if (e.empty()) e = r.Check(a.CreateTensorWithDataAsOrtValue(r.memory, (void*)target, px * sizeof(float), shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[1]));
        if (e.empty()) e = r.Check(a.CreateTensorWithDataAsOrtValue(r.memory, &patchWeight, sizeof(float), one, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[2]));
        static const char* inNames[3] = { "render", "target", "patch_weight" };
        static const char* outNames[3] = { "loss", "grad_render", "grad_target" };
        if (e.empty()) e = r.Check(a.Run(r.session, nullptr, inNames, in, 3, outNames, 3, out));
        float* results[3] = { loss, gradRender, gradTarget };
        const size_t sizes[3] = { 1, px, px };
        for (int k = 0; k < 3 && e.empty(); k++) {
            if (!results[k]) continue;
            void* data = nullptr;
            e = r.Check(a.GetTensorMutableData(out[k], &data));
            if (e.empty()) memcpy(results[k], data, sizes[k] * sizeof(float));
        }
        for (int k = 0; k < 3; k++) {
            if (in[k]) a.ReleaseValue(in[k]);
            if (out[k]) a.ReleaseValue(out[k]);
        }
        if (!e.empty()) m_Error = "DINOv3 failed: " + e;
        return e.empty();
    }
}
