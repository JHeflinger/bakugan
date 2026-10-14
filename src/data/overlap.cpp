// OpenVDB / TBB first: raylib's macros (PI, EPSILON, ...) must not leak into their headers.
#include <openvdb/openvdb.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/LevelSetMeasure.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include "overlap.hpp"
#include <raymath.h>
#include <algorithm>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <exception>
#include <future>
#include <mutex>

namespace Bakugan {
    uint64_t MeshData::NextId() {
        static std::atomic<uint64_t> next{ 1 };
        return next++;
    }

    namespace {
        using Clock = std::chrono::steady_clock;

        void InitOpenVDB() {
            static std::once_flag once;
            std::call_once(once, [] { openvdb::initialize(); });
        }

        // A level set of one mesh, kept between jobs while the mesh and resolution stay the same.
        struct CachedSDF {
            uint64_t meshId = 0;
            int resolution = 0;
            openvdb::FloatGrid::Ptr grid;
            float voxel = 0.0f;
            double volume = 0.0;
        };

        struct Cache {
            CachedSDF source, target;
        };

        float LongestSide(const BoundingBox& b) {
            return fmaxf(b.max.x - b.min.x, fmaxf(b.max.y - b.min.y, b.max.z - b.min.z));
        }

        const CachedSDF& EnsureSDF(CachedSDF& slot, const MeshData& mesh, int resolution) {
            if (slot.grid && slot.meshId == mesh.id && slot.resolution == resolution) return slot;
            float side = LongestSide(mesh.box);
            float voxel = side > 1e-9f ? side / (float)resolution : 1.0f / (float)resolution;
            openvdb::math::Transform::Ptr xform = openvdb::math::Transform::createLinearTransform(voxel);
            std::vector<openvdb::Vec3s> points(mesh.vertices.size());
            for (size_t i = 0; i < points.size(); i++)
                points[i] = openvdb::Vec3s(mesh.vertices[i].x, mesh.vertices[i].y, mesh.vertices[i].z);
            std::vector<openvdb::Vec3I> tris(mesh.triangles.size() / 3);
            for (size_t i = 0; i < tris.size(); i++)
                tris[i] = openvdb::Vec3I(mesh.triangles[i * 3], mesh.triangles[i * 3 + 1], mesh.triangles[i * 3 + 2]);
            // The sign comes from a flood fill, so this needs a closed mesh. Duplicated vertices (as in
            // raylib's primitives) are fine: voxelisation only looks at where the triangles are.
            slot.grid = openvdb::tools::meshToLevelSet<openvdb::FloatGrid>(*xform, points, tris, 3.0f);
            slot.meshId = mesh.id;
            slot.resolution = resolution;
            slot.voxel = voxel;
            slot.volume = openvdb::tools::levelSetVolume(*slot.grid);
            return slot;
        }

        struct Segment {
            Vector3 a, b;     // bone ends in the left mesh's rest space
            Matrix toRest;    // right world space -> rest space for the bone that carries this segment
            Matrix toPosed;   // and back: rest space -> right world space
        };

        inline float DistanceSquaredToSegment(Vector3 p, const Segment& s) {
            Vector3 ab = Vector3Subtract(s.b, s.a);
            float len2 = Vector3DotProduct(ab, ab);
            float t = len2 > 1e-12f ? Clamp(Vector3DotProduct(Vector3Subtract(p, s.a), ab) / len2, 0.0f, 1.0f) : 0.0f;
            Vector3 q = Vector3Add(s.a, Vector3Scale(ab, t));
            Vector3 d = Vector3Subtract(p, q);
            return Vector3DotProduct(d, d);
        }

        OverlapResult ComputeWithCache(const OverlapRequest& rq, Cache& cache) {
            auto start = Clock::now();
            OverlapResult out;
            out.signature = rq.signature;
            try {
                InitOpenVDB();
                if (!rq.source || !rq.target || rq.source->triangles.empty() || rq.target->triangles.empty()) {
                    out.message = "Both panels need a model.";
                    return out;
                }
                size_t joints = rq.parent.size();
                if (joints == 0 || rq.restWorld.size() != joints || rq.posedWorld.size() != joints) {
                    out.message = "The skeleton is empty.";
                    return out;
                }
                const int res = std::clamp(rq.params.resolution, 16, 256);
                const CachedSDF& src = EnsureSDF(cache.source, *rq.source, res);
                const CachedSDF& dst = EnsureSDF(cache.target, *rq.target, res);
                if (dst.volume <= 1e-12) {
                    out.message = "The right model isn't a closed mesh, so it has no inside to highlight.";
                    return out;
                }
                if (src.volume <= 1e-12) {
                    out.message = "The left model isn't a closed mesh, so its volume can't be folded.";
                    return out;
                }

                // One segment per bone (parent joint -> child joint). A bone is carried by its parent joint's frame.
                std::vector<Matrix> toRest(joints), toPosed(joints);
                for (size_t j = 0; j < joints; j++) {
                    Matrix move = MatrixMultiply(MatrixInvert(rq.restWorld[j]), rq.posedWorld[j]);   // rest -> posed (armature space)
                    toPosed[j] = MatrixMultiply(move, rq.place);                                      // rest -> right world
                    toRest[j] = MatrixInvert(toPosed[j]);                                             // right world -> rest
                }
                std::vector<Segment> segments;
                for (size_t j = 0; j < joints; j++) {
                    int p = rq.parent[j];
                    if (p < 0 || p >= (int)joints) continue;
                    Vector3 a = { rq.restWorld[p].m12, rq.restWorld[p].m13, rq.restWorld[p].m14 };
                    Vector3 b = { rq.restWorld[j].m12, rq.restWorld[j].m13, rq.restWorld[j].m14 };
                    segments.push_back({ a, b, toRest[p], toPosed[p] });
                }
                if (segments.empty()) {   // no bones yet: the whole mesh moves with the root
                    Vector3 a = { rq.restWorld[0].m12, rq.restWorld[0].m13, rq.restWorld[0].m14 };
                    segments.push_back({ a, a, toRest[0], toPosed[0] });
                }

                const float solidBelow = -fmaxf(rq.params.erosion, 0.0f) * src.voxel;   // level set value that counts as solid
                const openvdb::FloatGrid& dstGrid = *dst.grid;
                const openvdb::FloatGrid& srcGrid = *src.grid;

                auto nearestSegment = [&](Vector3 x) {
                    int owner = 0;
                    float best = FLT_MAX;
                    for (int o = 0; o < (int)segments.size(); o++) {
                        float d = DistanceSquaredToSegment(x, segments[o]);
                        if (d < best) { best = d; owner = o; }
                    }
                    return owner;
                };

                // Output box: the right mesh plus wherever the folded left mesh ends up, since parts of it can
                // stick out of the right mesh.
                BoundingBox region = rq.target->box;
                for (const Vector3& v : rq.source->vertices) {
                    Vector3 p = Vector3Transform(v, segments[nearestSegment(v)].toPosed);
                    region.min = Vector3Min(region.min, p);
                    region.max = Vector3Max(region.max, p);
                }
                const float vs = LongestSide(region) / (float)res;
                openvdb::Coord lo((int)std::floor(region.min.x / vs) - 2, (int)std::floor(region.min.y / vs) - 2, (int)std::floor(region.min.z / vs) - 2);
                openvdb::Coord hi((int)std::ceil(region.max.x / vs) + 2, (int)std::ceil(region.max.y / vs) + 2, (int)std::ceil(region.max.z / vs) + 2);
                const int nx = hi.x() - lo.x() + 1, ny = hi.y() - lo.y() + 1, nz = hi.z() - lo.z() + 1;
                out.dim[0] = nx; out.dim[1] = ny; out.dim[2] = nz;
                out.voxel = vs;
                out.origin = { lo.x() * vs, lo.y() * vs, lo.z() * vs };
                out.cells.assign((size_t)nx * ny * nz, 0);
                uint8_t* cells = out.cells.data();

                // Accessors cache tree paths and aren't thread safe, so each task owns a set.
                struct Probe {
                    openvdb::FloatGrid::ConstAccessor dstAcc, srcAcc;
                    openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler> dstSampler, srcSampler;
                    Probe(const openvdb::FloatGrid& d, const openvdb::FloatGrid& sgrid)
                        : dstAcc(d.getConstAccessor()), srcAcc(sgrid.getConstAccessor()),
                          dstSampler(dstAcc, d.transform()), srcSampler(srcAcc, sgrid.transform()) {}
                    bool InsideTarget(Vector3 p) { return dstSampler.wsSample(openvdb::Vec3d(p.x, p.y, p.z)) < 0.0f; }
                };
                // How many bones' folded parts cover a point (stops counting at 2).
                auto hitsAt = [&](Probe& pr, Vector3 p) {
                    int hits = 0;
                    for (int s = 0; s < (int)segments.size() && hits < 2; s++) {
                        Vector3 x = Vector3Transform(p, segments[s].toRest);     // where this point came from
                        if (pr.srcSampler.wsSample(openvdb::Vec3d(x.x, x.y, x.z)) >= solidBelow) continue;   // not solid there
                        if (nearestSegment(x) == s) hits++;                       // and that bone owns the spot
                    }
                    return hits;
                };

                // Right-side cells: 1 green (folded mesh fills it), 2 red (parts overlap inside the right mesh),
                // 3 yellow (inside the right mesh but empty), 4 red (folded mesh sticks out of the right mesh).
                tbb::parallel_for(tbb::blocked_range<int>(0, nz), [&](const tbb::blocked_range<int>& range) {
                    Probe pr(dstGrid, srcGrid);
                    for (int k = range.begin(); k < range.end(); k++)
                        for (int j = 0; j < ny; j++)
                            for (int i = 0; i < nx; i++) {
                                Vector3 p = { (lo.x() + i) * vs, (lo.y() + j) * vs, (lo.z() + k) * vs };
                                bool inside = pr.InsideTarget(p);
                                int hits = hitsAt(pr, p);
                                uint8_t c = 0;
                                if (hits == 0) c = inside ? 3 : 0;
                                else if (!inside) c = 4;
                                else c = hits >= 2 ? 2 : 1;
                                cells[((size_t)k * ny + j) * nx + i] = c;
                            }
                });

                // Left-side cells: every solid voxel is green, or red if its folded position overlaps another
                // part or lies outside the right mesh.
                const float svs = src.voxel;
                const BoundingBox& sb = rq.source->box;
                openvdb::Coord slo((int)std::floor(sb.min.x / svs) - 2, (int)std::floor(sb.min.y / svs) - 2, (int)std::floor(sb.min.z / svs) - 2);
                openvdb::Coord shi((int)std::ceil(sb.max.x / svs) + 2, (int)std::ceil(sb.max.y / svs) + 2, (int)std::ceil(sb.max.z / svs) + 2);
                const int sx = shi.x() - slo.x() + 1, sy = shi.y() - slo.y() + 1, sz = shi.z() - slo.z() + 1;
                out.sourceDim[0] = sx; out.sourceDim[1] = sy; out.sourceDim[2] = sz;
                out.sourceVoxel = svs;
                out.sourceOrigin = { slo.x() * svs, slo.y() * svs, slo.z() * svs };
                out.sourceCells.assign((size_t)sx * sy * sz, 0);
                uint8_t* scells = out.sourceCells.data();
                tbb::parallel_for(tbb::blocked_range<int>(0, sz), [&](const tbb::blocked_range<int>& range) {
                    Probe pr(dstGrid, srcGrid);
                    for (int k = range.begin(); k < range.end(); k++)
                        for (int j = 0; j < sy; j++)
                            for (int i = 0; i < sx; i++) {
                                openvdb::Coord ijk(slo.x() + i, slo.y() + j, slo.z() + k);
                                if (pr.srcAcc.getValue(ijk) >= 0.0f) continue;                 // not part of the left mesh
                                Vector3 x = { ijk.x() * svs, ijk.y() * svs, ijk.z() * svs };
                                Vector3 p = Vector3Transform(x, segments[nearestSegment(x)].toPosed);   // where the fold puts it
                                bool bad = !pr.InsideTarget(p) || hitsAt(pr, p) >= 2;
                                scells[((size_t)k * sy + j) * sx + i] = bad ? 2 : 1;
                            }
                });

                size_t n[5] = { 0, 0, 0, 0, 0 };
                for (uint8_t c : out.cells) n[c]++;
                const double voxelVolume = (double)vs * vs * vs;
                const size_t folded = n[1] + n[2] + n[4];          // voxels the folded mesh occupies
                const size_t shape = n[1] + n[2] + n[3];           // voxels of the right mesh
                out.targetVolume = dst.volume;
                out.overlapVolume = (n[2] + n[4]) * voxelVolume;
                out.ok = true;
                char buf[240];
                snprintf(buf, sizeof(buf), "%.1f%% of the folded mesh is red (%.1f%% overlap inside, %.1f%% outside the right shape). %.1f%% of the right shape is empty.",
                         folded ? 100.0 * (n[2] + n[4]) / folded : 0.0, folded ? 100.0 * n[2] / folded : 0.0,
                         folded ? 100.0 * n[4] / folded : 0.0, shape ? 100.0 * n[3] / shape : 0.0);
                out.message = buf;
            } catch (const std::exception& e) {
                out.ok = false;
                out.message = std::string("Overlap analysis failed: ") + e.what();
            }
            out.milliseconds = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            return out;
        }
    }

    struct OverlapField::Impl {
        // Declaration order matters: the future's destructor waits for a running job,
        // so it has to be destroyed before the cache that job is using.
        Cache cache;
        bool running = false;
        bool hasPending = false;
        OverlapRequest pending;
        std::future<OverlapResult> job;

        void Launch(OverlapRequest request) {
            running = true;
            job = std::async(std::launch::async, [this, request = std::move(request)]() {
                return ComputeWithCache(request, cache);
            });
        }
    };

    OverlapField::OverlapField() : m_Impl(new Impl) {}
    OverlapField::~OverlapField() = default;

    void OverlapField::Request(OverlapRequest request) {
        if (m_Impl->running) {
            m_Impl->pending = std::move(request);   // only the newest waiting request matters
            m_Impl->hasPending = true;
        } else {
            m_Impl->Launch(std::move(request));
        }
    }

    bool OverlapField::Poll(OverlapResult* out) {
        if (!m_Impl->running) return false;
        if (m_Impl->job.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;
        *out = m_Impl->job.get();
        m_Impl->running = false;
        if (m_Impl->hasPending) {
            m_Impl->hasPending = false;
            m_Impl->Launch(std::move(m_Impl->pending));
        }
        return true;
    }

    bool OverlapField::Busy() const { return m_Impl->running; }

    OverlapResult OverlapField::Compute(const OverlapRequest& request) {
        Cache cache;
        return ComputeWithCache(request, cache);
    }
}
