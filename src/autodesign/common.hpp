#pragma once
// Small shared pieces of the auto designer: random numbers, box and rotation helpers, and a thread pool.
#include "data/design.hpp"
#include "util/box.hpp"
#include <raymath.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

namespace Bakugan::Auto {
    using Rng = std::mt19937;
    using Clock = std::chrono::steady_clock;

    // ---- random numbers ---------------------------------------------------------------------------------------------

    inline float Uniform(Rng& r, float a = 0.0f, float b = 1.0f) { return std::uniform_real_distribution<float>(a, b)(r); }
    inline float Normal(Rng& r, float s = 1.0f) { return std::normal_distribution<float>(0.0f, s)(r); }
    inline int Pick(Rng& r, int n) { return n <= 1 ? 0 : std::uniform_int_distribution<int>(0, n - 1)(r); }
    inline bool Chance(Rng& r, float p) { return Uniform(r) < p; }
    template <class T> const T& PickFrom(Rng& r, const std::vector<T>& v) { return v[Pick(r, (int)v.size())]; }
    Vector3 RandomUnit(Rng& r);                  // a direction
    Vector3 RandomInBall(Rng& r);                // a point in the unit ball
    Quaternion RandomRotation(Rng& r);

    // ---- geometry ---------------------------------------------------------------------------------------------------

    inline float Coord(const Vector3& v, int i) { return (&v.x)[i]; }
    inline float& Coord(Vector3& v, int i) { return (&v.x)[i]; }
    inline float MinOf(Vector3 v) { return fminf(v.x, fminf(v.y, v.z)); }
    inline float MaxOf(Vector3 v) { return fmaxf(v.x, fmaxf(v.y, v.z)); }
    float WrapAngle(float a);                    // into (-pi, pi]
    float BoxDistance(Vector3 p, const BoundingBox& b);     // 0 inside
    Vector3 InBox(Vector3 lo, Vector3 ext, Vector3 unit);   // unit in [0, 1]^3 mapped into the box
    Quaternion FromAxes(Vector3 x, Vector3 y, Vector3 z);   // rotation turning x, y, z onto the given axes
    // An oriented box around a point cloud: local z along its longest principal direction, extents trimmed by `trim`
    // at each end, never thinner than minSize.
    SuperFrustum BoxAround(const std::vector<Vector3>& points, float trim, float minSize);

    // ---- a small thread pool: a score is thousands of independent field evaluations ---------------------------------

    class Pool {
    public:
        explicit Pool(int workers);
        ~Pool();
        Pool(const Pool&) = delete;
        Pool& operator=(const Pool&) = delete;

        // Calls f(0) .. f(n - 1), spread over the workers and the calling thread; returns when all are done.
        void For(int n, const std::function<void(int)>& f);
        // The same for many cheap calls: handed out in runs of 128.
        void ForEach(int n, const std::function<void(int)>& f);
    private:
        void Drain();
        void Work();
    private:
        std::vector<std::thread> m_Threads;
        std::mutex m_Mutex;
        std::condition_variable m_Wake, m_Done;
        const std::function<void(int)>* m_Job = nullptr;
        int m_Count = 0;
        std::atomic<int> m_Next{ 0 };
        int m_Finished = 0;
        int m_Generation = 0;
        bool m_Quit = false;
    };
}
