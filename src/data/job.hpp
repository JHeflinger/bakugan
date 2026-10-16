#pragma once
#include "data/design.hpp"
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace Bakugan {
    // Shared between a worker and the UI: progress, cancellation, and a live preview of the design being optimised.
    class JobControl {
    public:
        bool Cancelled() const { return m_Cancel.load(); }
        void Cancel() { m_Cancel = true; }
        void SetProgress(float f) { m_Progress = f; }
        float Progress() const { return m_Progress.load(); }
        void SetStage(const std::string& s) { std::lock_guard<std::mutex> l(m_Mutex); m_Stage = s; }
        std::string Stage() const { std::lock_guard<std::mutex> l(m_Mutex); return m_Stage; }
        void Publish(const Design& d) {
            std::lock_guard<std::mutex> l(m_Mutex);
            m_Preview = std::make_shared<const Design>(d);
            m_PreviewVersion++;
        }
        std::shared_ptr<const Design> Preview(uint64_t* version) const {
            std::lock_guard<std::mutex> l(m_Mutex);
            if (version) *version = m_PreviewVersion;
            return m_Preview;
        }
        void PushLoss(float v) { std::lock_guard<std::mutex> l(m_Mutex); m_History.push_back(v); }
        std::vector<float> History() const { std::lock_guard<std::mutex> l(m_Mutex); return m_History; }
    private:
        std::atomic<bool> m_Cancel{ false };
        std::atomic<float> m_Progress{ 0.0f };
        mutable std::mutex m_Mutex;
        std::string m_Stage;
        std::shared_ptr<const Design> m_Preview;
        uint64_t m_PreviewVersion = 0;
        std::vector<float> m_History;
    };

    // Runs one function at a time on a worker thread and hands its result back to the UI thread.
    template <class Result>
    class Job {
    public:
        Job() = default;
        Job(const Job&) = delete;
        Job& operator=(const Job&) = delete;
        ~Job() { Stop(); }

        void Start(std::function<Result(JobControl&)> fn) {
            Stop();
            m_Control = std::make_shared<JobControl>();
            m_Done = false;
            m_Running = true;
            auto control = m_Control;
            m_Thread = std::thread([this, fn, control] {
                Result r = fn(*control);
                std::lock_guard<std::mutex> l(m_Mutex);
                m_Result = std::move(r);
                m_Done = true;
            });
        }

        // True once, when a finished result was moved into *out.
        bool Poll(Result* out) {
            if (!m_Running) return false;
            {
                std::lock_guard<std::mutex> l(m_Mutex);
                if (!m_Done) return false;
                *out = std::move(m_Result);
            }
            m_Thread.join();
            m_Running = false;
            return true;
        }

        bool Busy() const { return m_Running; }
        void Cancel() { if (m_Control) m_Control->Cancel(); }
        const JobControl* Control() const { return m_Control.get(); }

        void Stop() {
            if (m_Thread.joinable()) {
                if (m_Control) m_Control->Cancel();
                m_Thread.join();
            }
            m_Running = false;
        }
    private:
        std::thread m_Thread;
        std::mutex m_Mutex;
        std::shared_ptr<JobControl> m_Control;
        Result m_Result{};
        bool m_Done = false;
        bool m_Running = false;
    };
}
