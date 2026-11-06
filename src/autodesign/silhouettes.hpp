#pragma once
// The Silhouettes fit: outlines of the design compared with outlines of the mesh and of the sphere, from several
// directions.
#include "autodesign.hpp"
#include "probe.hpp"
#include <unordered_map>

namespace Bakugan::Auto {
    // One orthographic view: looking along `dir`, with `u` to the right and `v` up in the picture.
    struct SilView {
        Vector3 dir = { 0, 0, -1 }, u = { 1, 0, 0 }, v = { 0, 1, 0 };
    };

    // Every view frames the same square, `half` wide either side of `centre` (2·half across, `res` pixels).
    struct SilFrame {
        Vector3 centre = { 0, 0, 0 };
        float half = 1.0f;
        int res = 64;
        std::vector<SilView> views;
        float Pixel() const { return 2.0f * half / res; }
        int Pixels() const { return res * res; }
        // where a world point lands in a view, in pixels (x right, y down)
        Vector2 ToPixel(const SilView& w, Vector3 p) const {
            Vector3 q = Vector3Subtract(p, centre);
            return { (Vector3DotProduct(q, w.u) / half + 1.0f) * 0.5f * res, (1.0f - Vector3DotProduct(q, w.v) / half) * 0.5f * res };
        }
    };

    // The design's outlines in every view: one byte of coverage per pixel, view after view.
    struct Outlines {
        std::vector<unsigned char> open, closed;
    };

    class Silhouettes {
    public:
        // Frames the mesh and draws its outline in every view, once.
        void Prepare(const TriMesh& mesh, const BoundingBox& box, float L, int views, int res, Pool& pool);
        bool Ready() const { return !m_Frame.views.empty(); }
        bool OutsideFrame(Vector3 p) const { return Vector3Distance(p, m_Frame.centre) > m_Frame.half; }

        // The outline terms of the score: fit, sphere, view coverage, the minimum-coverage penalty (see the .cpp).
        // offOpen / offClosed are the shares of material outside the frame in each pose.
        void Score(const Design& d, const Probe& pr, const std::vector<Rigid>& open, const std::vector<Rigid>& closed,
                   float offOpen, float offClosed, float sphereBias, float minCoverage, AutoScore* sc);
        const Outlines& Last() const { return m_Last; }       // of the design scored last

        // Outlines against their targets as one RGBA picture: open pose above, closed below.
        void Image(const Outlines& o, const AutoScore& s, std::vector<unsigned char>* rgba, int* w, int* h) const;
    private:
        // Rendered body outlines (every view, one pose), kept by what was rendered: the body's contents and where the
        // pose puts it. A descent step usually changes one body, so the others are found here, not traced again.
        class Cache {
        public:
            void Limit(size_t entryBytes);
            std::vector<unsigned char>* Find(unsigned long long key);
            std::vector<unsigned char>* Add(unsigned long long key, size_t bytes);
        private:
            struct Entry {
                std::vector<unsigned char> pixels;
                unsigned long long used = 0;
            };
            std::unordered_map<unsigned long long, Entry> m_Map;
            unsigned long long m_Tick = 0;
            size_t m_Cap = 256;
        };
        void Trace(const Design& d, const Probe& pr, const std::vector<Rigid>& open, const std::vector<Rigid>& closed);
    private:
        Pool* m_Pool = nullptr;
        SilFrame m_Frame;
        std::vector<std::vector<float>> m_Target;     // the mesh's outline in each view (pixel coverage 0..1)
        Cache m_Cache;
        Outlines m_Last;
    };
}
