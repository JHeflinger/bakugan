#include "mesh.h"
#include "util.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void mesh_free(TriMesh* m) {
    free(m->pos);
    free(m->idx);
    memset(m, 0, sizeof(*m));
}

TriMesh mesh_copy(const TriMesh* m) {
    TriMesh c = { 0 };
    c.nv = m->nv;
    c.nt = m->nt;
    c.pos = FB_NEW(V3, m->nv > 0 ? m->nv : 1);
    c.idx = FB_NEW(int, m->nt > 0 ? 3 * m->nt : 1);
    if (m->nv) memcpy(c.pos, m->pos, sizeof(V3) * (size_t)m->nv);
    if (m->nt) memcpy(c.idx, m->idx, sizeof(int) * 3 * (size_t)m->nt);
    return c;
}

bool mesh_load_obj(const char* path, TriMesh* out, char* err, int errSize) {
    memset(out, 0, sizeof(*out));
    size_t size;
    char* text = fb_read_file(path, &size);
    if (!text) { snprintf(err, (size_t)errSize, "can't read %s", path); return false; }
    FB_ARRAY(V3) pos = { 0 };
    FB_ARRAY(int) idx = { 0 };
    char* line = text;
    while (line && *line) {
        char* next = strchr(line, '\n');
        if (next) *next++ = 0;
        while (*line == ' ' || *line == '\t') line++;
        if (line[0] == 'v' && (line[1] == ' ' || line[1] == '\t')) {
            V3 p = { 0, 0, 0 };
            sscanf(line + 2, "%f %f %f", &p.x, &p.y, &p.z);
            fb_push(&pos, p);
        } else if (line[0] == 'f' && (line[1] == ' ' || line[1] == '\t')) {
            // a polygon: v, v/vt, v/vt/vn or v//vn; negative indices count back; fanned into triangles
            int poly[64], n = 0;
            char* c = line + 2;
            while (*c && n < 64) {
                while (*c == ' ' || *c == '\t' || *c == '\r') c++;
                if (!*c) break;
                char* e;
                long v = strtol(c, &e, 10);
                if (e == c) break;
                if (v < 0) v = pos.count + v + 1;
                poly[n++] = (int)v - 1;
                c = e;
                while (*c && *c != ' ' && *c != '\t') c++;
            }
            for (int k = 1; k + 1 < n; k++) {
                fb_push(&idx, poly[0]);
                fb_push(&idx, poly[k]);
                fb_push(&idx, poly[k + 1]);
            }
        }
        line = next;
    }
    free(text);
    for (int k = 0; k < idx.count; k++)
        if (idx.data[k] < 0 || idx.data[k] >= pos.count) {
            snprintf(err, (size_t)errSize, "%s: a face refers to vertex %d, which isn't there", path, idx.data[k] + 1);
            fb_free_array(&pos);
            fb_free_array(&idx);
            return false;
        }
    if (idx.count == 0) {
        snprintf(err, (size_t)errSize, "%s has no faces", path);
        fb_free_array(&pos);
        fb_free_array(&idx);
        return false;
    }
    out->pos = pos.data;
    out->nv = pos.count;
    out->idx = idx.data;
    out->nt = idx.count / 3;
    return true;
}

bool mesh_save_obj(const char* path, const TriMesh* m) {
    FILE* f = fopen(path, "w");
    if (!f) return false;
    for (int i = 0; i < m->nv; i++) fprintf(f, "v %.6f %.6f %.6f\n", m->pos[i].x, m->pos[i].y, m->pos[i].z);
    for (int t = 0; t < m->nt; t++) fprintf(f, "f %d %d %d\n", m->idx[3 * t] + 1, m->idx[3 * t + 1] + 1, m->idx[3 * t + 2] + 1);
    return fclose(f) == 0;
}

Box3 mesh_bounds(const TriMesh* m) {
    Box3 b = box_empty();
    for (int i = 0; i < m->nv; i++) b = box_add(b, m->pos[i]);
    return b;
}

double mesh_volume(const TriMesh* m) {
    double v = 0;
    for (int t = 0; t < m->nt; t++) {
        V3 a = m->pos[m->idx[3 * t]], b = m->pos[m->idx[3 * t + 1]], c = m->pos[m->idx[3 * t + 2]];
        v += v3dot(a, v3cross(b, c)) / 6.0;
    }
    return v;
}

float mesh_normalize(TriMesh* m, float size) {
    Box3 b = mesh_bounds(m);
    V3 c = v3scale(v3add(b.min, b.max), 0.5f), e = v3sub(b.max, b.min);
    float longest = fb_max(e.x, fb_max(e.y, e.z));
    float s = longest > 1e-12f ? size / longest : 1.0f;
    for (int i = 0; i < m->nv; i++) m->pos[i] = v3scale(v3sub(m->pos[i], c), s);
    return s;
}

// ---- welding and simplifying (both by snapping vertices to cells and merging what lands together)

typedef struct { long long key; int id; } CellKey;
static int cmp_cell(const void* a, const void* b) {
    long long x = ((const CellKey*)a)->key, y = ((const CellKey*)b)->key;
    return x < y ? -1 : (x > y);
}

static void cluster(TriMesh* m, float cell, bool average) {
    if (m->nv == 0) return;
    Box3 b = mesh_bounds(m);
    CellKey* keys = FB_NEW(CellKey, m->nv);
    for (int i = 0; i < m->nv; i++) {
        V3 q = v3sub(m->pos[i], b.min);
        long long x = (long long)floorf(q.x / cell), y = (long long)floorf(q.y / cell), z = (long long)floorf(q.z / cell);
        keys[i].key = (x * 2097152LL + y) * 2097152LL + z;
        keys[i].id = i;
    }
    qsort(keys, (size_t)m->nv, sizeof(CellKey), cmp_cell);
    int* remap = FB_NEW(int, m->nv);
    V3* pos = FB_NEW(V3, m->nv);
    int n = 0;
    for (int s = 0; s < m->nv;) {
        int e = s;
        V3 sum = v3(0, 0, 0);
        while (e < m->nv && keys[e].key == keys[s].key) { sum = v3add(sum, m->pos[keys[e].id]); e++; }
        pos[n] = average ? v3scale(sum, 1.0f / (e - s)) : m->pos[keys[s].id];
        for (int k = s; k < e; k++) remap[keys[k].id] = n;
        n++;
        s = e;
    }
    // triangles: remapped, degenerate ones dropped
    int nt = 0;
    for (int t = 0; t < m->nt; t++) {
        int a = remap[m->idx[3 * t]], bb = remap[m->idx[3 * t + 1]], c = remap[m->idx[3 * t + 2]];
        if (a == bb || bb == c || a == c) continue;
        m->idx[3 * nt] = a;
        m->idx[3 * nt + 1] = bb;
        m->idx[3 * nt + 2] = c;
        nt++;
    }
    free(m->pos);
    m->pos = pos;
    m->nv = n;
    m->nt = nt;
    free(remap);
    free(keys);
}

void mesh_weld(TriMesh* m, float eps) { cluster(m, eps, false); }

TriMesh mesh_simplify(const TriMesh* m, int target) {
    TriMesh c = mesh_copy(m);
    if (m->nv <= target) return c;
    Box3 b = mesh_bounds(m);
    V3 e = v3sub(b.max, b.min);
    // a cell size giving about `target` occupied cells (surface ~ area / cell^2)
    double area = 0;
    for (int t = 0; t < m->nt; t++) {
        V3 p = m->pos[m->idx[3 * t]], q = m->pos[m->idx[3 * t + 1]], r = m->pos[m->idx[3 * t + 2]];
        area += 0.5 * v3len(v3cross(v3sub(q, p), v3sub(r, p)));
    }
    float cell = (float)sqrt(area / target) * 0.9f;
    float longest = fb_max(e.x, fb_max(e.y, e.z));
    if (cell < longest * 1e-4f) cell = longest * 1e-4f;
    for (int tries = 0; tries < 8; tries++) {
        TriMesh d = mesh_copy(m);
        cluster(&d, cell, true);
        if (d.nv <= target * 1.3f || tries == 7) {
            mesh_free(&c);
            return d;
        }
        mesh_free(&d);
        cell *= 1.25f;
    }
    return c;
}

// ---- views

View view_make(V3 dir, V3 center, float half, int res) {
    View w;
    w.dir = v3norm(dir);
    V3 up = fabsf(w.dir.y) < 0.95f ? v3(0, 1, 0) : v3(0, 0, 1);
    w.u = v3norm(v3cross(up, w.dir));
    w.v = v3cross(w.dir, w.u);
    w.center = center;
    w.half = half;
    w.res = res;
    return w;
}

void view_directions(int n, V3* out) {
    // a Fibonacci spiral over the upper half sphere (no view looks back along another), tilted so no view is
    // exactly along an axis (meshes are full of axis aligned faces)
    for (int k = 0; k < n; k++) {
        float z = 1.0f - (k + 0.5f) / n * 0.92f, r = sqrtf(fb_max(0.0f, 1.0f - z * z)), a = k * 2.39996323f + 0.3f;
        V3 d = v3(r * cosf(a), z, r * sinf(a));
        Quat tilt = qaxis(v3(1, 0.3f, 0.2f), 0.35f);
        out[k] = v3norm(qrot(tilt, d));
    }
}

void mesh_raster(const TriMesh* m, const View* w, int ss, float* cover, float* depth) {
    const int R = w->res * ss;
    float* z = FB_NEW(float, (size_t)R * R);
    for (int k = 0; k < R * R; k++) z[k] = FLT_MAX;
    const float scale = R / (2.0f * w->half);
    for (int t = 0; t < m->nt; t++) {
        float px[3], py[3], pz[3];
        for (int c = 0; c < 3; c++) {
            V3 q = v3sub(m->pos[m->idx[3 * t + c]], w->center);
            px[c] = (v3dot(q, w->u) + w->half) * scale;
            py[c] = (w->half - v3dot(q, w->v)) * scale;
            pz[c] = v3dot(q, w->dir);
        }
        float area = (px[1] - px[0]) * (py[2] - py[0]) - (py[1] - py[0]) * (px[2] - px[0]);
        if (fabsf(area) < 1e-12f) continue;
        int x0 = fb_maxi(0, (int)floorf(fb_min(px[0], fb_min(px[1], px[2])))), x1 = fb_mini(R - 1, (int)ceilf(fb_max(px[0], fb_max(px[1], px[2]))));
        int y0 = fb_maxi(0, (int)floorf(fb_min(py[0], fb_min(py[1], py[2])))), y1 = fb_mini(R - 1, (int)ceilf(fb_max(py[0], fb_max(py[1], py[2]))));
        for (int y = y0; y <= y1; y++)
            for (int x = x0; x <= x1; x++) {
                float sx = x + 0.5f, sy = y + 0.5f;
                float a = ((px[1] - sx) * (py[2] - sy) - (py[1] - sy) * (px[2] - sx)) / area;
                float b = ((px[2] - sx) * (py[0] - sy) - (py[2] - sy) * (px[0] - sx)) / area;
                float c = 1.0f - a - b;
                if (a < 0 || b < 0 || c < 0) continue;
                float d = a * pz[0] + b * pz[1] + c * pz[2];
                float* zz = &z[(size_t)y * R + x];
                if (d < *zz) *zz = d;
            }
    }
    const int res = w->res;
    for (int j = 0; j < res; j++)
        for (int i = 0; i < res; i++) {
            int hit = 0;
            double zs = 0;
            for (int b = 0; b < ss; b++)
                for (int a = 0; a < ss; a++) {
                    float d = z[(size_t)(j * ss + b) * R + i * ss + a];
                    if (d < FLT_MAX) { hit++; zs += d; }
                }
            cover[j * res + i] = (float)hit / (ss * ss);
            if (depth) depth[j * res + i] = hit ? (float)(zs / hit) : 0.0f;
        }
    free(z);
}
