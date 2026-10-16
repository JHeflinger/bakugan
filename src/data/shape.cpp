// OpenVDB / TBB first: raylib's macros (PI, EPSILON, ...) must not leak into their headers.
#include <openvdb/openvdb.h>
#include <openvdb/tools/MeshToVolume.h>
#include "shape.hpp"
#include <raymath.h>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <random>

namespace Bakugan {
    static void InitOpenVDB() {
        static std::once_flag once;
        std::call_once(once, [] { openvdb::initialize(); });
    }

    float ShapeField::Sample(Vector3 p) const {
        if (sdf.empty()) return 1e9f;
        float fx = (p.x - origin.x) / voxel, fy = (p.y - origin.y) / voxel, fz = (p.z - origin.z) / voxel;
        float ox = fmaxf(fmaxf(-fx, fx - (dim[0] - 1)), 0.0f);
        float oy = fmaxf(fmaxf(-fy, fy - (dim[1] - 1)), 0.0f);
        float oz = fmaxf(fmaxf(-fz, fz - (dim[2] - 1)), 0.0f);
        float outside = sqrtf(ox * ox + oy * oy + oz * oz) * voxel;
        fx = std::clamp(fx, 0.0f, (float)dim[0] - 1.001f);
        fy = std::clamp(fy, 0.0f, (float)dim[1] - 1.001f);
        fz = std::clamp(fz, 0.0f, (float)dim[2] - 1.001f);
        int i = (int)fx, j = (int)fy, k = (int)fz;
        float tx = fx - i, ty = fy - j, tz = fz - k;
        auto at = [&](int a, int b, int c) { return sdf[Index(a, b, c)]; };
        float c00 = at(i, j, k) * (1 - tx) + at(i + 1, j, k) * tx;
        float c10 = at(i, j + 1, k) * (1 - tx) + at(i + 1, j + 1, k) * tx;
        float c01 = at(i, j, k + 1) * (1 - tx) + at(i + 1, j, k + 1) * tx;
        float c11 = at(i, j + 1, k + 1) * (1 - tx) + at(i + 1, j + 1, k + 1) * tx;
        float v = (c00 * (1 - ty) + c10 * ty) * (1 - tz) + (c01 * (1 - ty) + c11 * ty) * tz;
        return outside > 0 ? fmaxf(v, 0.0f) + outside : v;
    }

    float ShapeField::Laplacian(Vector3 p) const {
        float h = voxel, c = Sample(p);
        float s = Sample({ p.x + h, p.y, p.z }) + Sample({ p.x - h, p.y, p.z }) + Sample({ p.x, p.y + h, p.z }) +
                  Sample({ p.x, p.y - h, p.z }) + Sample({ p.x, p.y, p.z + h }) + Sample({ p.x, p.y, p.z - h });
        return (s - 6.0f * c) / (h * h);
    }

    ShapeField BuildShapeField(const TriMesh& mesh, float voxel, float padding, std::string* err) {
        ShapeField f;
        if (mesh.indices.size() < 3 || voxel <= 0.0f) { *err = "Empty mesh"; return f; }
        InitOpenVDB();
        try {
            openvdb::math::Transform::Ptr xform = openvdb::math::Transform::createLinearTransform(voxel);
            std::vector<openvdb::Vec3s> points(mesh.positions.size());
            for (size_t i = 0; i < points.size(); i++)
                points[i] = openvdb::Vec3s(mesh.positions[i].x, mesh.positions[i].y, mesh.positions[i].z);
            std::vector<openvdb::Vec3I> tris(mesh.indices.size() / 3);
            for (size_t i = 0; i < tris.size(); i++)
                tris[i] = openvdb::Vec3I(mesh.indices[i * 3], mesh.indices[i * 3 + 1], mesh.indices[i * 3 + 2]);
            std::vector<openvdb::Vec4I> quads;

            Vector3 size = Vector3Subtract(mesh.box.max, mesh.box.min);
            int maxDim = (int)ceilf(fmaxf(size.x, fmaxf(size.y, size.z)) / voxel) + 8;
            float exBand = fmaxf(3.0f, padding / voxel + 2.0f);
            openvdb::FloatGrid::Ptr grid = openvdb::tools::meshToSignedDistanceField<openvdb::FloatGrid>(
                *xform, points, tris, quads, exBand, (float)maxDim);

            // dense copy over the padded mesh box, voxel centres aligned with the vdb index space
            int lo[3], hi[3];
            for (int a = 0; a < 3; a++) {
                lo[a] = (int)floorf(((&mesh.box.min.x)[a] - padding) / voxel);
                hi[a] = (int)ceilf(((&mesh.box.max.x)[a] + padding) / voxel);
                f.dim[a] = hi[a] - lo[a] + 1;
            }
            f.voxel = voxel;
            f.origin = { lo[0] * voxel, lo[1] * voxel, lo[2] * voxel };
            f.sdf.resize((size_t)f.dim[0] * f.dim[1] * f.dim[2]);
            openvdb::FloatGrid::ConstAccessor acc = grid->getConstAccessor();
            for (int k = 0; k < f.dim[2]; k++)
                for (int j = 0; j < f.dim[1]; j++)
                    for (int i = 0; i < f.dim[0]; i++)
                        f.sdf[f.Index(i, j, k)] = acc.getValue(openvdb::Coord(lo[0] + i, lo[1] + j, lo[2] + k));
        } catch (const std::exception& e) {
            *err = std::string("OpenVDB: ") + e.what();
            f.sdf.clear();
        }
        return f;
    }

    ShapeAnalysis AnalyzeShape(const TriMesh& mesh, int resolution, int detailSamples) {
        ShapeAnalysis a;
        Vector3 size = Vector3Subtract(mesh.box.max, mesh.box.min);
        float longest = fmaxf(size.x, fmaxf(size.y, size.z));
        if (mesh.indices.empty() || longest <= 0) { a.message = "No geometry"; return a; }
        float voxel = longest / (float)std::max(resolution, 8);
        std::string err;
        a.field = BuildShapeField(mesh, voxel, voxel * 3.0f, &err);
        if (!a.field.Valid()) { a.message = err; return a; }

        const ShapeField& f = a.field;
        Vector3 sum = { 0, 0, 0 };
        for (int k = 0; k < f.dim[2]; k++)
            for (int j = 0; j < f.dim[1]; j++)
                for (int i = 0; i < f.dim[0]; i++)
                    if (f.sdf[f.Index(i, j, k)] < 0.0f) {
                        Vector3 p = f.Position(i, j, k);
                        a.interior.push_back(p);
                        sum = Vector3Add(sum, p);
                    }
        if (a.interior.empty()) { a.message = "The mesh has no inside (is it closed?)"; return a; }
        a.volume = (double)a.interior.size() * voxel * voxel * voxel;
        a.centroid = Vector3Scale(sum, 1.0f / a.interior.size());
        a.equalRadius = (float)cbrt(3.0 * a.volume / (4.0 * PI));

        // area weighted surface samples, weighted up where the surface bends sharply
        std::vector<double> cdf;
        double total = 0;
        for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
            Vector3 p0 = mesh.positions[mesh.indices[t]], p1 = mesh.positions[mesh.indices[t + 1]], p2 = mesh.positions[mesh.indices[t + 2]];
            total += 0.5 * Vector3Length(Vector3CrossProduct(Vector3Subtract(p1, p0), Vector3Subtract(p2, p0)));
            cdf.push_back(total);
        }
        std::mt19937 rng(1234);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        for (int s = 0; s < detailSamples && total > 0; s++) {
            size_t t = std::lower_bound(cdf.begin(), cdf.end(), U(rng) * total) - cdf.begin();
            t = std::min(t, cdf.size() - 1);
            float r1 = sqrtf((float)U(rng)), r2 = (float)U(rng);
            Vector3 p0 = mesh.positions[mesh.indices[t * 3]], p1 = mesh.positions[mesh.indices[t * 3 + 1]], p2 = mesh.positions[mesh.indices[t * 3 + 2]];
            Vector3 p = Vector3Add(Vector3Add(Vector3Scale(p0, 1 - r1), Vector3Scale(p1, r1 * (1 - r2))), Vector3Scale(p2, r1 * r2));
            float curvature = fabsf(f.Laplacian(p)) * voxel;
            a.detail.push_back({ p, 1.0f + 3.0f * std::clamp(curvature, 0.0f, 1.5f) });
        }
        a.ok = true;
        a.message = "Volume " + std::to_string(a.volume).substr(0, 6) + ", " + std::to_string(a.interior.size()) + " voxels";
        return a;
    }
}
