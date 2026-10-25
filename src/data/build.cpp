// OpenVDB / TBB first: raylib's macros (PI, EPSILON, ...) must not leak into their headers.
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/SignedFloodFill.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include "build.hpp"
#include "util/utils.hpp"
#include <raymath.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <sstream>

namespace Bakugan {
    namespace {
        void InitVDB() {
            static std::once_flag once;
            std::call_once(once, [] { openvdb::initialize(); });
        }

        BoundingBox TransformBox(BoundingBox b, const Rigid& r) {
            Vector3 lo = { 1e30f, 1e30f, 1e30f }, hi = { -1e30f, -1e30f, -1e30f };
            for (int i = 0; i < 8; i++) {
                Vector3 c = { (i & 1) ? b.max.x : b.min.x, (i & 2) ? b.max.y : b.min.y, (i & 4) ? b.max.z : b.min.z };
                Vector3 w = r.Apply(c);
                lo = Vector3Min(lo, w);
                hi = Vector3Max(hi, w);
            }
            return { lo, hi };
        }

        bool Overlap(BoundingBox a, BoundingBox b) {
            return a.min.x <= b.max.x && a.max.x >= b.min.x && a.min.y <= b.max.y && a.max.y >= b.min.y && a.min.z <= b.max.z &&
                   a.max.z >= b.min.z;
        }

        bool InBox(Vector3 p, BoundingBox b) {
            return p.x >= b.min.x && p.x <= b.max.x && p.y >= b.min.y && p.y <= b.max.y && p.z >= b.min.z && p.z <= b.max.z;
        }

        BoundingBox Grow(BoundingBox b, float m) {
            return { Vector3SubtractValue(b.min, m), Vector3AddValue(b.max, m) };
        }

        std::string FileSafe(const std::string& s) {
            std::string out;
            for (char ch : s) {
                bool ok = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '-' || ch == '_';
                out += ok ? ch : '_';
            }
            return out.empty() ? "part" : out;
        }
    }

    struct LevelSet {
        openvdb::FloatGrid::Ptr grid;
        float voxel = 1.0f;
    };

    struct LevelSetProbe::Impl {
        openvdb::FloatGrid::ConstAccessor acc;
        const openvdb::math::Transform& xform;
        float voxel;
        Impl(const LevelSet& ls) : acc(ls.grid->getConstAccessor()), xform(ls.grid->transform()), voxel(ls.voxel) {}
    };

    LevelSetProbe::LevelSetProbe(const LevelSet& ls) : m(new Impl(ls)) {}
    LevelSetProbe::~LevelSetProbe() { delete m; }
    float LevelSetProbe::Voxel() const { return m->voxel; }

    float LevelSetProbe::operator()(Vector3 p) {
        openvdb::Vec3d ijk = m->xform.worldToIndex(openvdb::Vec3d(p.x, p.y, p.z));
        return openvdb::tools::BoxSampler::sample(m->acc, ijk);
    }

    TriMesh MeshLevelSet(const std::function<float(Vector3)>& f, BoundingBox box, float voxel, float adaptivity,
                         const JobControl* control, const std::function<void(float)>& progress, std::shared_ptr<const LevelSet>* keep) {
        InitVDB();
        TriMesh out;
        const float band = 3.0f * voxel;
        const int pad = 3, B = 4, T = 64;   // 4^3 leaf blocks: finer culling of empty space
        // the odd fraction keeps grid points off the round-number planes designs are full of: samples exactly on a
        // face give the mesher ties that come out as pinched (non-manifold) edges
        const Vector3 jitter = { 0.3713f, 0.2847f, 0.4129f };
        Vector3 origin = Vector3Subtract(box.min, { (pad + jitter.x) * voxel, (pad + jitter.y) * voxel, (pad + jitter.z) * voxel });
        int n[3];
        for (int a = 0; a < 3; a++) n[a] = (int)ceilf(((&box.max.x)[a] - (&box.min.x)[a]) / voxel) + 2 * pad + 2;
        int nt[3] = { (n[0] + T - 1) / T, (n[1] + T - 1) / T, (n[2] + T - 1) / T };
        size_t tiles = (size_t)nt[0] * nt[1] * nt[2];
        auto at = [&](float i, float j, float k) { return Vector3{ origin.x + i * voxel, origin.y + j * voxel, origin.z + k * voxel }; };

        using Grid = openvdb::FloatGrid;
        tbb::enumerable_thread_specific<Grid::Ptr> local([band] { return Grid::create(band); });
        std::atomic<size_t> done{ 0 };

        // Octree descent: a cube of s voxels is skipped when the field says the surface cannot pass through it.
        std::function<void(Grid::Accessor&, int, int, int, int)> visit = [&](Grid::Accessor& acc, int i0, int j0, int k0, int s) {
            if (i0 >= n[0] || j0 >= n[1] || k0 >= n[2]) return;
            float half = s * 0.5f;
            float d = f(at(i0 + half, j0 + half, k0 + half));
            float reach = 1.7320508f * half * voxel;
            if (fabsf(d) > reach * 1.25f + band) return;
            if (s <= B) {
                for (int k = k0; k < std::min(k0 + s, n[2]); k++)
                    for (int j = j0; j < std::min(j0 + s, n[1]); j++)
                        for (int i = i0; i < std::min(i0 + s, n[0]); i++) {
                            float v = f(at((float)i, (float)j, (float)k));
                            if (fabsf(v) < band) acc.setValue(openvdb::Coord(i, j, k), v);
                        }
                return;
            }
            int h = s / 2;
            for (int c = 0; c < 8; c++) visit(acc, i0 + (c & 1) * h, j0 + ((c >> 1) & 1) * h, k0 + ((c >> 2) & 1) * h, h);
        };

        tbb::parallel_for(tbb::blocked_range<size_t>(0, tiles, 1), [&](const tbb::blocked_range<size_t>& r) {
            Grid::Ptr& g = local.local();
            Grid::Accessor acc = g->getAccessor();
            for (size_t t = r.begin(); t != r.end(); t++) {
                if (control && control->Cancelled()) return;
                int ti = (int)(t % nt[0]), tj = (int)((t / nt[0]) % nt[1]), tk = (int)(t / ((size_t)nt[0] * nt[1]));
                visit(acc, ti * T, tj * T, tk * T, T);
                size_t k = ++done;
                if (progress && (k & 15) == 0) progress(0.85f * k / (float)tiles);
            }
        });
        if (control && control->Cancelled()) return out;

        Grid::Ptr grid = Grid::create(band);
        for (Grid::Ptr& g : local) grid->tree().merge(g->tree(), openvdb::MERGE_ACTIVE_STATES);
        if (grid->activeVoxelCount() == 0) return out;
        openvdb::math::Transform::Ptr xform = openvdb::math::Transform::createLinearTransform(voxel);
        xform->postTranslate(openvdb::Vec3d(origin.x, origin.y, origin.z));
        grid->setTransform(xform);
        grid->setGridClass(openvdb::GRID_LEVEL_SET);
        openvdb::tools::signedFloodFill(grid->tree());
        if (keep) {
            auto ls = std::make_shared<LevelSet>();
            ls->grid = grid;
            ls->voxel = voxel;
            *keep = ls;
        }

        std::vector<openvdb::Vec3s> points;
        std::vector<openvdb::Vec3I> tris;
        std::vector<openvdb::Vec4I> quads;
        openvdb::tools::volumeToMesh(*grid, points, tris, quads, 0.0, adaptivity);
        out.positions.resize(points.size());
        for (size_t i = 0; i < points.size(); i++) out.positions[i] = { points[i].x(), points[i].y(), points[i].z() };
        for (const openvdb::Vec3I& t : tris) out.indices.insert(out.indices.end(), { t[0], t[1], t[2] });
        for (const openvdb::Vec4I& q : quads) out.indices.insert(out.indices.end(), { q[0], q[1], q[2], q[0], q[2], q[3] });
        // make the winding face outward (positive signed volume)
        double vol = 0;
        for (size_t t = 0; t + 2 < out.indices.size(); t += 3) {
            Vector3 a = out.positions[out.indices[t]], b = out.positions[out.indices[t + 1]], c = out.positions[out.indices[t + 2]];
            vol += Vector3DotProduct(a, Vector3CrossProduct(b, c));
        }
        if (vol < 0)
            for (size_t t = 0; t + 2 < out.indices.size(); t += 3) std::swap(out.indices[t + 1], out.indices[t + 2]);
        Utils::CleanMesh(out);
        Utils::ComputeNormals(out);
        Utils::ComputeBounds(out);
        if (progress) progress(1.0f);
        return out;
    }

    float PreviewVoxel(const Design& d, int resolution) {
        BoundingBox b = d.Bounds();
        for (const Hinge& h : d.hinges) {
            float r = h.length * 0.5f + h.radius;
            b.min = Vector3Min(b.min, Vector3SubtractValue(h.pivot, r));
            b.max = Vector3Max(b.max, Vector3AddValue(h.pivot, r));
        }
        Vector3 s = Vector3Subtract(b.max, b.min);
        float longest = fmaxf(s.x, fmaxf(s.y, s.z));
        return Clamp(longest / (float)std::max(resolution, 16), 0.12f, 4.0f);
    }

    static void PutHingeHardware(std::ostringstream& s, const Design& d, int h) {
        const Hinge& hg = d.hinges[h];
        Vector3 la = d.LeafDirection(h, false), lb = d.LeafDirection(h, true);
        s << "h" << hg.id << ' ' << hg.parent << ' ' << hg.child << ' ' << hg.pivot.x << ' ' << hg.pivot.y << ' ' << hg.pivot.z << ' '
          << hg.axis.x << ' ' << hg.axis.y << ' ' << hg.axis.z << ' ' << hg.hardware << ' ' << hg.length << ' ' << hg.radius << ' '
          << hg.pinRadius << ' ' << hg.clearance << ' ' << hg.knuckles << ' ' << (int)hg.pin << ' ' << hg.leafLength << ' '
          << hg.leafThickness << ' ' << la.x << ' ' << la.y << ' ' << la.z << ' ' << lb.x << ' ' << lb.y << ' ' << lb.z << ';';
    }

    std::string PartKey(const Design& d, int bodyId) {
        std::ostringstream s;
        s.precision(9);
        s << "body" << bodyId << ';';
        for (const Group& g : d.groups)
            if (d.Inside(g.id, bodyId)) s << "g" << g.id << ' ' << g.parent << ' ' << (int)g.op << ' ' << g.blend << ';';
        for (const Shape& sh : d.shapes) {
            if (!d.Inside(sh.group, bodyId)) continue;
            const SuperFrustum& f = sh.sf;
            s << "s" << sh.id << ' ' << sh.group << ' ' << (int)sh.op << ' ' << f.position.x << ' ' << f.position.y << ' ' << f.position.z << ' '
              << f.rotation.x << ' ' << f.rotation.y << ' ' << f.rotation.z << ' ' << f.rotation.w << ' ' << f.size.x << ' ' << f.size.y << ' '
              << f.size.z << ' ' << f.rounding << ' ' << f.taper << ' ' << f.bulge << ' ' << f.dilation << ' ' << f.onion << ';';
        }
        for (int h = 0; h < (int)d.hinges.size(); h++)
            if (d.hinges[h].parent == bodyId || d.hinges[h].child == bodyId) PutHingeHardware(s, d, h);
        return s.str();
    }

    std::string PinKey(const Design& d, int h) {
        std::ostringstream s;
        s.precision(9);
        s << "pin;";
        PutHingeHardware(s, d, h);
        return s.str();
    }

    BuildResult BuildParts(const Design& d, float voxel, float adaptivity, int samples, JobControl& c, const std::vector<Rigid>* pose,
                           float gap, const std::vector<const BuiltPart*>* reuse) {
        BuildResult r;
        r.design = d;
        r.voxel = voxel;
        std::vector<BodyField> fields;
        std::vector<std::string> names, keys;
        std::vector<int> groupIndex;
        char settings[96];
        snprintf(settings, sizeof(settings), "|v%.6g a%.4g n%d", voxel, adaptivity, samples);
        for (int g : d.Bodies()) {
            BodyField f = BodyField::Body(d, d.groups[g].id);
            if (f.Empty()) continue;
            fields.push_back(std::move(f));
            names.push_back(d.groups[g].name);
            keys.push_back(PartKey(d, d.groups[g].id) + settings);
            groupIndex.push_back(g);
        }
        size_t bodies = fields.size();
        // the bodies each one touches (or nearly) in the print pose
        struct Neighbour { size_t j; Rigid rel; BoundingBox box; };
        std::vector<std::vector<Neighbour>> near(bodies);
        bool trimming = pose && gap > 0.0f;
        if (trimming)
            for (size_t i = 0; i < bodies; i++)
                for (size_t j = 0; j < bodies; j++) {
                    if (i == j) continue;
                    Rigid rel = (*pose)[groupIndex[i]].Then((*pose)[groupIndex[j]].Inverse());   // i's frame -> j's frame
                    BoundingBox box = Grow(fields[j].bounds, gap);
                    if (Overlap(TransformBox(fields[i].bounds, rel), box)) near[i].push_back({ j, rel, box });
                }
        for (int h = 0; h < (int)d.hinges.size(); h++) {
            if (!d.hinges[h].hardware || d.hinges[h].pin != PinStyle::Separate) continue;
            fields.push_back(BodyField::Pin(d, h));
            names.push_back(d.hinges[h].name + " pin");
            keys.push_back(PinKey(d, h) + settings);
        }
        if (fields.empty()) { r.ok = true; r.message = "Nothing to build."; return r; }
        // what can be carried over, and how much is left to mesh (for the progress bar)
        std::vector<const BuiltPart*> old(fields.size(), nullptr);
        if (reuse && !trimming)
            for (size_t i = 0; i < fields.size(); i++)
                for (const BuiltPart* p : *reuse)
                    if (p && p->key == keys[i]) { old[i] = p; break; }
        size_t todo = 0, done = 0;
        for (const BuiltPart* p : old) todo += p ? 0 : 1;
        for (size_t i = 0; i < fields.size(); i++) {
            if (old[i]) {
                BuiltPart part = *old[i];
                part.name = names[i];
                r.parts.push_back(std::move(part));
                continue;
            }
            const BodyField& f = fields[i];
            c.SetStage("Meshing " + names[i]);
            float base = (float)done / todo, span = 1.0f / todo;
            done++;
            r.rebuilt++;
            BuiltPart part;
            part.body = f.body;
            part.pinHinge = f.pinHinge;
            part.name = names[i];
            part.key = keys[i];
            const std::vector<Neighbour>* nb = i < bodies && !near[i].empty() ? &near[i] : nullptr;
            float half = gap * 0.5f;
            auto field = [&f, &fields, nb, half](Vector3 p) {
                float v = f.Distance(p);
                if (nb)
                    for (const Neighbour& n : *nb) {
                        Vector3 q = n.rel.Apply(p);
                        if (InBox(q, n.box)) v = fmaxf(v, half - fields[n.j].Distance(q));
                    }
                return v;
            };
            part.mesh = MeshLevelSet(field, f.bounds, voxel, adaptivity, &c, [&c, base, span](float x) { c.SetProgress(base + span * 0.9f * x); },
                                     samples > 0 ? &part.levelSet : nullptr);
            // when stopped, the parts finished so far still go back: the next build can reuse them
            if (c.Cancelled()) { r.cancelled = true; r.message = "Stopped."; return r; }
            if (samples > 0 && !part.mesh.positions.empty()) {
                size_t stride = std::max<size_t>(1, part.mesh.positions.size() / (size_t)samples);
                size_t count = (part.mesh.positions.size() + stride - 1) / stride;
                part.samples.resize(count);
                float h = voxel * 0.25f;
                const TriMesh& mesh = part.mesh;
                tbb::parallel_for(tbb::blocked_range<size_t>(0, count, 64), [&](const tbb::blocked_range<size_t>& range) {
                    if (c.Cancelled()) return;
                    for (size_t k = range.begin(); k != range.end(); k++) {
                        Vector3 p = mesh.positions[k * stride];
                        // Newton steps onto the exact surface; kept only if they really land closer (creases and thin
                        // walls can send the finite difference gradient astray)
                        Vector3 q = p;
                        float v0 = fabsf(f.Distance(p));
                        for (int it = 0; it < 3; it++) {
                            float v = f.Distance(q);
                            Vector3 g = f.Gradient(q, h);
                            float g2 = Vector3DotProduct(g, g);
                            if (g2 < 0.25f) break;
                            Vector3 step = Vector3Scale(g, v / g2);
                            if (Vector3Length(step) > voxel) break;
                            q = Vector3Subtract(q, step);
                        }
                        if (fabsf(f.Distance(q)) < v0 && Vector3Distance(p, q) < voxel * 1.5f) p = q;
                        part.samples[k] = p;
                    }
                });
                if (c.Cancelled()) { r.cancelled = true; r.message = "Stopped."; return r; }
            }
            r.parts.push_back(std::move(part));
        }
        c.SetProgress(1.0f);
        r.ok = true;
        r.message = "Built " + std::to_string(r.parts.size()) + " part" + (r.parts.size() == 1 ? "" : "s") + " (" +
                    std::to_string(r.rebuilt) + " remeshed).";
        return r;
    }

    static const char* PoseName(ExportPose p) {
        return p == ExportPose::Open ? "open" : p == ExportPose::Closed ? "closed" : "design";
    }

    ExportResult ExportSTL(const Design& d, const ExportParams& p, JobControl& c) {
        ExportResult r;
        std::vector<Rigid> xf = d.PoseTransforms(p.pose, p.ease);
        BuildResult b = BuildParts(d, p.voxel, p.adaptivity, 0, c, &xf, p.keepApart ? p.gap : 0.0f);
        if (!b.ok) { r.cancelled = b.cancelled; r.message = b.cancelled ? "Export stopped." : b.message; return r; }
        if (b.parts.empty()) { r.message = "Nothing to export: the design has no bodies."; return r; }
        c.SetStage("Writing STL files");
        auto place = [&](Vector3 v, const Rigid& t) {
            Vector3 w = Vector3Scale(t.Apply(v), p.scale);
            return p.zUp ? Vector3{ w.x, -w.z, w.y } : w;
        };
        std::vector<TriMesh> placed;
        for (BuiltPart& part : b.parts) {
            int g = d.GroupIndex(part.body);
            if (part.pinHinge >= 0) {
                int h = d.HingeIndex(part.pinHinge);
                g = h >= 0 ? d.GroupIndex(d.hinges[h].parent) : -1;
            }
            Rigid t = g >= 0 ? xf[g] : Rigid{};
            TriMesh m = std::move(part.mesh);
            for (Vector3& v : m.positions) v = place(v, t);
            Utils::ComputeBounds(m);
            placed.push_back(std::move(m));
        }
        float lift = 0.0f;
        if (p.dropToBed) {
            float lo = 1e30f;
            for (const TriMesh& m : placed)
                if (!m.positions.empty()) lo = fminf(lo, p.zUp ? m.box.min.z : m.box.min.y);
            lift = lo < 1e29f ? -lo : 0.0f;
        }
        for (TriMesh& m : placed) {
            for (Vector3& v : m.positions) (p.zUp ? v.z : v.y) += lift;
            Utils::ComputeBounds(m);
        }

        std::string err;
        if (p.separate) {
            for (size_t i = 0; i < placed.size(); i++) {
                std::string path = p.prefix + "_" + FileSafe(b.parts[i].name) + ".stl";
                if (!Utils::WriteSTL(path, placed[i], &err)) { r.message = err; return r; }
                r.files.push_back(path);
            }
        }
        if (p.combined) {
            TriMesh all;
            for (const TriMesh& m : placed) {
                unsigned base = (unsigned)all.positions.size();
                all.positions.insert(all.positions.end(), m.positions.begin(), m.positions.end());
                for (unsigned i : m.indices) all.indices.push_back(base + i);
            }
            std::string path = p.prefix + "_assembly.stl";
            if (!Utils::WriteSTL(path, all, &err)) { r.message = err; return r; }
            r.files.push_back(path);
        }

        // a short assembly sheet next to the parts
        std::string notes = p.prefix + "_notes.txt";
        {
            std::string t;
            t += Utils::Format("Bakugan hinge designer - print sheet\n");
            t += Utils::Format("units: mm (x%.3g)   pose: %s   up axis: %s   voxel %.3f mm", p.scale, PoseName(p.pose), p.zUp ? "+Z" : "+Y", p.voxel);
            if (p.keepApart) t += Utils::Format("   bodies kept %.2f mm apart", p.gap);
            t += Utils::Format("\n\n");
            t += Utils::Format("%-28s %10s %10s %10s %10s\n", "part", "size x", "size y", "size z", "triangles");
            for (size_t i = 0; i < placed.size(); i++) {
                Vector3 s = Vector3Subtract(placed[i].box.max, placed[i].box.min);
                t += Utils::Format("%-28s %10.2f %10.2f %10.2f %10zu\n", b.parts[i].name.c_str(), s.x, s.y, s.z, placed[i].indices.size() / 3);
            }
            bool fused = false;
            if (!d.hinges.empty()) {
                t += Utils::Format("\n%-16s %-16s %-16s %8s %8s %4s %7s %6s %6s %s\n", "hinge", "parent", "child", "open", "closed", "kn",
                        "radius", "pin", "gap", "style");
                for (const Hinge& h : d.hinges) {
                    int pi = d.GroupIndex(h.parent), ci = d.GroupIndex(h.child);
                    t += Utils::Format("%-16s %-16s %-16s %7.1fd %7.1fd %4d %7.2f %6.2f %6.2f %s\n", h.name.c_str(),
                            pi >= 0 ? d.groups[pi].name.c_str() : "?", ci >= 0 ? d.groups[ci].name.c_str() : "?", h.start * RAD2DEG,
                            h.end * RAD2DEG, h.knuckles, h.radius, h.pinRadius, h.clearance,
                            !h.hardware ? "axis only (no hardware)" : h.pin == PinStyle::Fused ? "print in place" : "separate pin");
                    fused |= h.hardware && h.pin == PinStyle::Fused;
                }
            }
            t += Utils::Format("\nNotes\n");
            if (fused)
                t += Utils::Format("- Print-in-place hinges: the pin is part of the parent body and runs through the child's knuckles,\n"
                            "  so the joined bodies must be printed together, as exported (use the _assembly.stl). Break the\n"
                            "  hinge free gently after printing.\n");
            t += Utils::Format("- Separate-pin hinges: print the bodies and the pin parts on their own, line up the knuckles and push the\n"
                        "  pin in from the end without a head (print pins standing up, head down, for a round shaft).\n"
                        "- The gap is the designed clearance between moving surfaces. Below about 0.3 mm most FDM printers\n"
                        "  fuse the parts; raise it in the hinge settings if the hinge prints solid.\n");
            std::string err;
            if (Utils::WriteTextFile(notes, t, &err)) r.files.push_back(notes);
        }
        r.ok = true;
        r.message = "Exported " + std::to_string(placed.size()) + " part" + (placed.size() == 1 ? "" : "s") + " (" +
                    PoseName(p.pose) + " pose) to " + p.prefix + "_*.stl";
        return r;
    }
}
