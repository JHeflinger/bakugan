// OpenVDB / TBB first: raylib's macros (PI, EPSILON, ...) must not leak into their headers.
#include <openvdb/openvdb.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <tbb/parallel_for.h>
#include "restore.hpp"
#include "util/utils.hpp"
#include <raymath.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

namespace Bakugan {
    namespace {
        using Clock = std::chrono::steady_clock;

        struct Built {
            int part = -1;
            ShapeField field;          // final signed field of the piece, open pose
            BoundingBox closedBox;     // its box in the closed pose (world)
        };

        TriMesh MeshField(const ShapeField& f, float adaptivity) {
            TriMesh out;
            openvdb::initialize();
            openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(3.0f * f.voxel);
            openvdb::math::Transform::Ptr xform = openvdb::math::Transform::createLinearTransform(f.voxel);
            xform->postTranslate(openvdb::Vec3d(f.origin.x, f.origin.y, f.origin.z));
            grid->setTransform(xform);
            grid->setGridClass(openvdb::GRID_LEVEL_SET);
            openvdb::FloatGrid::Accessor acc = grid->getAccessor();
            for (int k = 0; k < f.dim[2]; k++)
                for (int j = 0; j < f.dim[1]; j++)
                    for (int i = 0; i < f.dim[0]; i++) {
                        float v = f.sdf[f.Index(i, j, k)];
                        if (v < 3.0f * f.voxel) acc.setValue(openvdb::Coord(i, j, k), v);
                    }
            std::vector<openvdb::Vec3s> points;
            std::vector<openvdb::Vec3I> tris;
            std::vector<openvdb::Vec4I> quads;
            openvdb::tools::volumeToMesh(*grid, points, tris, quads, 0.0, adaptivity);
            out.positions.resize(points.size());
            for (size_t i = 0; i < points.size(); i++) out.positions[i] = { points[i].x(), points[i].y(), points[i].z() };
            for (const openvdb::Vec3I& t : tris) out.indices.insert(out.indices.end(), { t[0], t[1], t[2] });
            for (const openvdb::Vec4I& q : quads)
                out.indices.insert(out.indices.end(), { q[0], q[1], q[2], q[0], q[2], q[3] });
            // make the winding face outward (positive signed volume)
            double vol = 0;
            for (size_t t = 0; t + 2 < out.indices.size(); t += 3) {
                Vector3 a = out.positions[out.indices[t]], b = out.positions[out.indices[t + 1]], c = out.positions[out.indices[t + 2]];
                vol += Vector3DotProduct(a, Vector3CrossProduct(b, c));
            }
            if (vol < 0)
                for (size_t t = 0; t + 2 < out.indices.size(); t += 3) std::swap(out.indices[t + 1], out.indices[t + 2]);
            Utils::ComputeNormals(out);
            Utils::ComputeBounds(out);
            return out;
        }

        BoundingBox Intersect(BoundingBox a, BoundingBox b) {
            return { Vector3Max(a.min, b.min), Vector3Min(a.max, b.max) };
        }

        BoundingBox Union(BoundingBox a, BoundingBox b) {
            return { Vector3Min(a.min, b.min), Vector3Max(a.max, b.max) };
        }

        BoundingBox Grow(BoundingBox b, float r) {
            return { Vector3Subtract(b.min, { r, r, r }), Vector3Add(b.max, { r, r, r }) };
        }

        bool EmptyBox(BoundingBox b) { return b.min.x >= b.max.x || b.min.y >= b.max.y || b.min.z >= b.max.z; }

        float Capsule(Vector3 p, Vector3 a, Vector3 b, float r) {
            Vector3 pa = Vector3Subtract(p, a), ba = Vector3Subtract(b, a);
            float bb = Vector3DotProduct(ba, ba);
            float h = bb > 1e-12f ? Clamp(Vector3DotProduct(pa, ba) / bb, 0.0f, 1.0f) : 0.0f;
            return Vector3Length(Vector3Subtract(pa, Vector3Scale(ba, h))) - r;
        }

        bool InBox(BoundingBox b, Vector3 p) {
            return p.x >= b.min.x && p.y >= b.min.y && p.z >= b.min.z && p.x <= b.max.x && p.y <= b.max.y && p.z <= b.max.z;
        }
    }

    RestoreResult RestoreDetail(const Design& d, const std::shared_ptr<const TriMesh>& mesh, const RestoreParams& prm,
                                JobControl& control) {
        RestoreResult result;
        result.design = d;
        auto t0 = Clock::now();
        if (!mesh || mesh->Empty()) { result.message = "Load a model first."; return result; }
        int n = (int)d.parts.size();
        if (n == 0) { result.message = "There are no parts to build."; return result; }
        const Vector3 c = d.ball.center;
        const float R = d.ball.radius;
        const float voxel = 2.0f * R / (float)std::max(prm.resolution, 16);
        const float shell = std::clamp(prm.shellThickness, 0.01f, 1.0f) * R;
        const float gap = prm.fillGaps ? prm.gapFill * R : 0.0f;

        control.SetStage("Level set of the input");
        std::string err;
        ShapeField material = BuildShapeField(*mesh, voxel, 3.0f * voxel, &err);
        if (!material.Valid()) { result.message = err; return result; }

        std::vector<Rigid> xf = d.Transforms(1.0f);
        std::vector<Vector3> openCentre(n), closedCentre(n);
        std::vector<float> reach(n);
        for (int i = 0; i < n; i++) {
            openCentre[i] = d.parts[i].shape.position;
            closedCentre[i] = xf[i].Apply(openCentre[i]);
            reach[i] = d.parts[i].shape.BoundingRadius();
        }

        // priority: the root, then bigger parts first
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        int root = d.Root();
        auto volume = [&](int i) { const Vector3& s = d.parts[i].shape.size; return s.x * s.y * s.z; };
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            if ((a == root) != (b == root)) return a == root;
            return volume(a) > volume(b);
        });

        std::vector<Built> built;
        for (int oi = 0; oi < n && !control.Cancelled(); oi++) {
            int i = order[oi];
            control.SetStage("Building " + d.parts[i].name + " (" + std::to_string(oi + 1) + "/" + std::to_string(n) + ")");
            const SuperFrustum& sf = d.parts[i].shape;
            // structural spars: from this part's hinge to its centre and on to each child's hinge, so a chain of
            // parts stays physically linked even where little of its material survives
            std::vector<std::pair<Vector3, Vector3>> spars;
            if (d.parts[i].hinge.parent >= 0) spars.push_back({ d.parts[i].hinge.pivot, sf.position });
            for (int ch : d.Children(i)) spars.push_back({ sf.position, d.parts[ch].hinge.pivot });
            float sparRadius = fmaxf(0.16f * fminf(sf.size.x, fminf(sf.size.y, sf.size.z)), 1.6f * voxel);
            Vector3 ballOpen = xf[i].ApplyInverse(c);
            BoundingBox ballBox = { Vector3Subtract(ballOpen, { R, R, R }), Vector3Add(ballOpen, { R, R, R }) };
            BoundingBox stuff = Union(mesh->box, Grow(sf.WorldBounds(), gap + voxel));
            for (const auto& [a, e] : spars) {
                stuff = Union(stuff, Grow({ Vector3Min(a, e), Vector3Max(a, e) }, sparRadius + voxel));
            }
            BoundingBox box = Grow(Intersect(ballBox, stuff), 2.0f * voxel);
            if (EmptyBox(box)) continue;

            Built b;
            b.part = i;
            ShapeField& f = b.field;
            f.voxel = voxel;
            f.origin = box.min;
            for (int a = 0; a < 3; a++) f.dim[a] = std::max(2, (int)ceilf(((&box.max.x)[a] - (&box.min.x)[a]) / voxel) + 1);
            f.sdf.assign((size_t)f.dim[0] * f.dim[1] * f.dim[2], 3.0f * voxel);

            tbb::parallel_for(0, f.dim[2], [&](int k) {
                for (int jj = 0; jj < f.dim[1]; jj++)
                    for (int ii = 0; ii < f.dim[0]; ii++) {
                        Vector3 x = f.Position(ii, jj, k);
                        Vector3 y = xf[i].Apply(x);
                        float r = Vector3Distance(y, c);
                        float dball = r - R;
                        if (dball > 2.0f * voxel) { f.sdf[f.Index(ii, jj, k)] = dball; continue; }
                        float own = sf.Distance(x);

                        // the part's cell of the input material (nearest superfrustum in the open pose)
                        float other = 1e30f;
                        for (int j = 0; j < n; j++) {
                            if (j == i) continue;
                            float lower = Vector3Distance(x, openCentre[j]) - reach[j] - d.parts[j].shape.dilation;
                            if (lower > fminf(other, own + 4.0f * voxel)) continue;
                            other = fminf(other, d.parts[j].shape.Distance(x));
                        }
                        float cell = other > 1e29f ? -1e3f : 0.5f * (own - other);
                        float dmat = fmaxf(material.Sample(x), cell);
                        for (const auto& [a, e] : spars) dmat = fminf(dmat, Capsule(x, a, e, sparRadius));

                        // the part's share of the sphere shell in the closed pose
                        float band = fmaxf(R - shell - r, r - R);
                        float dshell = band;
                        if (band < 2.0f * voxel) {
                            float rival = 1e30f;
                            for (int j = 0; j < n; j++) {
                                if (j == i) continue;
                                float lower = Vector3Distance(y, closedCentre[j]) - reach[j];
                                if (lower > fminf(rival, own + gap + 4.0f * voxel)) continue;
                                rival = fminf(rival, d.parts[j].shape.Distance(xf[j].ApplyInverse(y)));
                            }
                            float claim = rival > 1e29f ? own - gap : fmaxf(0.5f * (own - rival), own - gap);
                            dshell = fmaxf(band, claim);
                        }
                        float v = fmaxf(fminf(dmat, dshell), dball);

                        // carve out what higher priority pieces already occupy in the closed pose
                        for (const Built& o : built) {
                            if (!InBox(o.closedBox, y)) continue;
                            float w = o.field.Sample(xf[o.part].ApplyInverse(y));
                            v = fmaxf(v, -w);
                        }
                        f.sdf[f.Index(ii, jj, k)] = v;
                    }
            });

            // closed-pose box for the collision tests of later parts
            Vector3 lo = { 1e30f, 1e30f, 1e30f }, hi = { -1e30f, -1e30f, -1e30f };
            for (int q = 0; q < 8; q++) {
                Vector3 corner = { (q & 1) ? box.max.x : box.min.x, (q & 2) ? box.max.y : box.min.y, (q & 4) ? box.max.z : box.min.z };
                Vector3 w = xf[i].Apply(corner);
                lo = Vector3Min(lo, w);
                hi = Vector3Max(hi, w);
            }
            b.closedBox = { lo, hi };

            RestoredPart rp;
            rp.partId = d.parts[i].id;
            size_t inside = 0;
            for (float v : f.sdf) inside += v < 0;
            rp.volume = (float)inside * voxel * voxel * voxel;
            if (inside > 0) {
                try {
                    rp.mesh = MeshField(f, prm.adaptivity);
                } catch (const std::exception& e) {
                    result.message = std::string("OpenVDB: ") + e.what();
                    return result;
                }
            }
            if (!rp.mesh.Empty()) result.parts.push_back(std::move(rp));
            built.push_back(std::move(b));
            control.SetProgress((oi + 1) / (float)n);
        }

        // how much of the outer sphere the finished pieces cover
        std::vector<Vector3> dirs = Utils::FibonacciSphere(800);
        int covered = 0;
        for (const Vector3& u : dirs) {
            Vector3 y = Vector3Add(c, Vector3Scale(u, R - 0.5f * shell));
            for (const Built& o : built)
                if (InBox(o.closedBox, y) && o.field.Sample(xf[o.part].ApplyInverse(y)) < 0) { covered++; break; }
        }
        result.shellCoverage = covered / (float)dirs.size();
        result.cancelled = control.Cancelled();
        result.ok = !result.parts.empty() && !result.cancelled;
        result.milliseconds = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        char buf[160];
        snprintf(buf, sizeof(buf), "Built %d pieces in %.1f s. Sphere surface %.0f%% closed.", (int)result.parts.size(),
                 result.milliseconds / 1000.0, result.shellCoverage * 100.0f);
        if (result.message.empty()) result.message = result.cancelled ? "Stopped." : buf;
        return result;
    }
}
