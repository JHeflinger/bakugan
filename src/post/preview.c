#include "preview.h"
#include "mesher.h"
#include "core/util.h"
#include <stdlib.h>
#include <string.h>

FbImage image_create(int w, int h, unsigned char grey) {
    FbImage im = { w, h, FB_NEW(unsigned char, (size_t)w * h * 3) };
    memset(im.rgb, grey, (size_t)w * h * 3);
    return im;
}

void image_free(FbImage* im) {
    free(im->rgb);
    im->rgb = NULL;
}

// ---- PNG (zlib stored blocks: no compression, no dependency)

static unsigned crc_table[256];
static void crc_init(void) {
    for (unsigned n = 0; n < 256; n++) {
        unsigned c = n;
        for (int k = 0; k < 8; k++) c = c & 1 ? 0xedb88320u ^ (c >> 1) : c >> 1;
        crc_table[n] = c;
    }
}
static unsigned crc(unsigned c, const unsigned char* b, size_t n) {
    c ^= 0xffffffffu;
    for (size_t i = 0; i < n; i++) c = crc_table[(c ^ b[i]) & 0xff] ^ (c >> 8);
    return c ^ 0xffffffffu;
}
static void put32(unsigned char* p, unsigned v) { p[0] = v >> 24; p[1] = v >> 16; p[2] = v >> 8; p[3] = v; }
static void chunk(FILE* f, const char* type, const unsigned char* data, size_t n) {
    unsigned char len[4];
    put32(len, (unsigned)n);
    fwrite(len, 1, 4, f);
    unsigned char* buf = FB_NEW(unsigned char, n + 4);
    memcpy(buf, type, 4);
    if (n) memcpy(buf + 4, data, n);
    fwrite(buf, 1, n + 4, f);
    unsigned char c[4];
    put32(c, crc(0, buf, n + 4));
    fwrite(c, 1, 4, f);
    free(buf);
}

bool png_write(const char* path, const FbImage* im) {
    if (!crc_table[1]) crc_init();
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    static const unsigned char sig[8] = { 137, 80, 78, 71, 13, 10, 26, 10 };
    fwrite(sig, 1, 8, f);
    unsigned char hdr[13];
    put32(hdr, (unsigned)im->w);
    put32(hdr + 4, (unsigned)im->h);
    hdr[8] = 8; hdr[9] = 2; hdr[10] = 0; hdr[11] = 0; hdr[12] = 0;
    chunk(f, "IHDR", hdr, 13);
    const size_t row = (size_t)im->w * 3 + 1, raw = row * im->h;
    unsigned char* data = FB_NEW(unsigned char, raw);
    for (int y = 0; y < im->h; y++) {
        data[y * row] = 0;
        memcpy(data + y * row + 1, im->rgb + (size_t)y * im->w * 3, (size_t)im->w * 3);
    }
    const size_t blocks = (raw + 65534) / 65535;
    unsigned char* z = FB_NEW(unsigned char, raw + blocks * 5 + 6);
    size_t zn = 0;
    z[zn++] = 0x78;
    z[zn++] = 0x01;
    unsigned a = 1, b = 0;
    for (size_t off = 0; off < raw || off == 0;) {
        size_t n = raw - off > 65535 ? 65535 : raw - off;
        z[zn++] = off + n >= raw ? 1 : 0;
        z[zn++] = (unsigned char)(n & 0xff);
        z[zn++] = (unsigned char)(n >> 8);
        z[zn++] = (unsigned char)(~n & 0xff);
        z[zn++] = (unsigned char)((~n >> 8) & 0xff);
        memcpy(z + zn, data + off, n);
        for (size_t i = 0; i < n; i++) {
            a = (a + data[off + i]) % 65521;
            b = (b + a) % 65521;
        }
        zn += n;
        off += n;
        if (n == 0) break;
    }
    put32(z + zn, (b << 16) | a);
    zn += 4;
    chunk(f, "IDAT", z, zn);
    chunk(f, "IEND", NULL, 0);
    free(data);
    free(z);
    return fclose(f) == 0;
}

// ---- a tiny 5x7 font for labels (digits, upper case, a few marks)

static const unsigned char kGlyphs[][7] = {
    /* ' ' */ { 0, 0, 0, 0, 0, 0, 0 }, /* 0 */ { 14, 17, 19, 21, 25, 17, 14 }, /* 1 */ { 4, 12, 4, 4, 4, 4, 14 }, /* 2 */ { 14, 17, 1, 2, 4, 8, 31 },
    /* 3 */ { 31, 2, 4, 2, 1, 17, 14 }, /* 4 */ { 2, 6, 10, 18, 31, 2, 2 }, /* 5 */ { 31, 16, 30, 1, 1, 17, 14 }, /* 6 */ { 6, 8, 16, 30, 17, 17, 14 },
    /* 7 */ { 31, 1, 2, 4, 8, 8, 8 }, /* 8 */ { 14, 17, 17, 14, 17, 17, 14 }, /* 9 */ { 14, 17, 17, 15, 1, 2, 12 }, /* . */ { 0, 0, 0, 0, 0, 12, 12 },
    /* A */ { 14, 17, 17, 31, 17, 17, 17 }, /* B */ { 30, 17, 17, 30, 17, 17, 30 }, /* C */ { 14, 17, 16, 16, 16, 17, 14 }, /* D */ { 28, 18, 17, 17, 17, 18, 28 },
    /* E */ { 31, 16, 16, 30, 16, 16, 31 }, /* F */ { 31, 16, 16, 30, 16, 16, 16 }, /* G */ { 14, 17, 16, 23, 17, 17, 15 }, /* H */ { 17, 17, 17, 31, 17, 17, 17 },
    /* I */ { 14, 4, 4, 4, 4, 4, 14 }, /* J */ { 7, 2, 2, 2, 2, 18, 12 }, /* K */ { 17, 18, 20, 24, 20, 18, 17 }, /* L */ { 16, 16, 16, 16, 16, 16, 31 },
    /* M */ { 17, 27, 21, 21, 17, 17, 17 }, /* N */ { 17, 17, 25, 21, 19, 17, 17 }, /* O */ { 14, 17, 17, 17, 17, 17, 14 }, /* P */ { 30, 17, 17, 30, 16, 16, 16 },
    /* Q */ { 14, 17, 17, 17, 21, 18, 13 }, /* R */ { 30, 17, 17, 30, 20, 18, 17 }, /* S */ { 15, 16, 16, 14, 1, 1, 30 }, /* T */ { 31, 4, 4, 4, 4, 4, 4 },
    /* U */ { 17, 17, 17, 17, 17, 17, 14 }, /* V */ { 17, 17, 17, 17, 17, 10, 4 }, /* W */ { 17, 17, 17, 21, 21, 21, 10 }, /* X */ { 17, 17, 10, 4, 10, 17, 17 },
    /* Y */ { 17, 17, 10, 4, 4, 4, 4 }, /* Z */ { 31, 1, 2, 4, 8, 16, 31 }, /* : */ { 0, 12, 12, 0, 12, 12, 0 }, /* / */ { 1, 1, 2, 4, 8, 16, 16 },
    /* - */ { 0, 0, 0, 31, 0, 0, 0 }, /* % */ { 24, 25, 2, 4, 8, 19, 3 }, /* _ */ { 0, 0, 0, 0, 0, 0, 31 }, /* , */ { 0, 0, 0, 0, 12, 4, 8 },
    /* ( */ { 2, 4, 8, 8, 8, 4, 2 }, /* ) */ { 8, 4, 2, 2, 2, 4, 8 }, /* + */ { 0, 4, 4, 31, 4, 4, 0 }, /* = */ { 0, 0, 31, 0, 31, 0, 0 },
    /* x */ { 0, 0, 17, 10, 4, 10, 17 },
};
static int glyph(char c) {
    if (c >= 'a' && c <= 'z') c = (char)(c - 32);
    if (c >= '0' && c <= '9') return 1 + (c - '0');
    if (c == '.') return 11;
    if (c >= 'A' && c <= 'Z') return 12 + (c - 'A');
    if (c == ':') return 38;
    if (c == '/') return 39;
    if (c == '-') return 40;
    if (c == '%') return 41;
    if (c == '_') return 42;
    if (c == ',') return 43;
    if (c == '(') return 44;
    if (c == ')') return 45;
    if (c == '+') return 46;
    if (c == '=') return 47;
    if (c == '*') return 48;
    return 0;
}

void image_text(FbImage* im, int x, int y, int scale, const char* s, unsigned char r, unsigned char g, unsigned char b) {
    for (; *s; s++, x += 6 * scale) {
        const unsigned char* gl = kGlyphs[glyph(*s)];
        for (int row = 0; row < 7; row++)
            for (int col = 0; col < 5; col++) {
                if (!(gl[row] & (16 >> col))) continue;
                for (int sy = 0; sy < scale; sy++)
                    for (int sx = 0; sx < scale; sx++) {
                        int px = x + col * scale + sx, py = y + row * scale + sy;
                        if (px < 0 || py < 0 || px >= im->w || py >= im->h) continue;
                        unsigned char* p = im->rgb + ((size_t)py * im->w + px) * 3;
                        p[0] = r; p[1] = g; p[2] = b;
                    }
            }
    }
}

// ---- drawing meshes

V3 body_color(int index) {
    static const V3 pal[] = { { 0.90f, 0.45f, 0.30f }, { 0.30f, 0.60f, 0.90f }, { 0.40f, 0.80f, 0.40f }, { 0.95f, 0.80f, 0.30f },
                              { 0.70f, 0.45f, 0.85f }, { 0.30f, 0.80f, 0.80f }, { 0.90f, 0.50f, 0.65f }, { 0.60f, 0.65f, 0.35f },
                              { 0.55f, 0.55f, 0.95f }, { 0.95f, 0.60f, 0.40f }, { 0.45f, 0.70f, 0.60f }, { 0.80f, 0.35f, 0.40f } };
    return pal[index % 12];
}

DesignMeshes design_meshes(const Design* d, int cellsAcross) {
    DesignMeshes m;
    m.n = d->nbodies;
    m.bodies = FB_NEW(TriMesh, m.n > 0 ? m.n : 1);
    float cell = 2.0f * d->R / cellsAcross;
    for (int b = 0; b < d->nbodies; b++) m.bodies[b] = mesh_body(d, b, cell, 2 * cellsAcross);
    return m;
}

void design_meshes_free(DesignMeshes* m) {
    for (int b = 0; b < m->n; b++) mesh_free(&m->bodies[b]);
    free(m->bodies);
    m->bodies = NULL;
    m->n = 0;
}

typedef struct {
    FbImage* im;
    float* z;
    int x0, y0, size;
    View w;
} Canvas;

static void draw_mesh(Canvas* c, const TriMesh* m, const Rigid* xf, V3 color, float alpha) {
    const V3 light = v3norm(v3(-0.4f, 0.8f, -0.45f));
    const float scale = c->size / (2.0f * c->w.half);
    for (int t = 0; t < m->nt; t++) {
        V3 P[3];
        for (int k = 0; k < 3; k++) P[k] = xf ? rapply(*xf, m->pos[m->idx[3 * t + k]]) : m->pos[m->idx[3 * t + k]];
        V3 n = v3norm(v3cross(v3sub(P[1], P[0]), v3sub(P[2], P[0])));
        float shade = 0.25f + 0.75f * fb_max(0.0f, -v3dot(n, rrot(rident(), light)) * 0.5f + 0.5f * fabsf(v3dot(n, c->w.dir)));
        float px[3], py[3], pz[3];
        for (int k = 0; k < 3; k++) {
            V3 q = v3sub(P[k], c->w.center);
            px[k] = (v3dot(q, c->w.u) + c->w.half) * scale;
            py[k] = (c->w.half - v3dot(q, c->w.v)) * scale;
            pz[k] = v3dot(q, c->w.dir);
        }
        float area = (px[1] - px[0]) * (py[2] - py[0]) - (py[1] - py[0]) * (px[2] - px[0]);
        if (fabsf(area) < 1e-9f) continue;
        int xa = fb_maxi(0, (int)floorf(fb_min(px[0], fb_min(px[1], px[2])))), xb = fb_mini(c->size - 1, (int)ceilf(fb_max(px[0], fb_max(px[1], px[2]))));
        int ya = fb_maxi(0, (int)floorf(fb_min(py[0], fb_min(py[1], py[2])))), yb = fb_mini(c->size - 1, (int)ceilf(fb_max(py[0], fb_max(py[1], py[2]))));
        for (int y = ya; y <= yb; y++)
            for (int x = xa; x <= xb; x++) {
                float sx = x + 0.5f, sy = y + 0.5f;
                float a = ((px[1] - sx) * (py[2] - sy) - (py[1] - sy) * (px[2] - sx)) / area;
                float b = ((px[2] - sx) * (py[0] - sy) - (py[2] - sy) * (px[0] - sx)) / area;
                float cc = 1 - a - b;
                if (a < 0 || b < 0 || cc < 0) continue;
                float depth = a * pz[0] + b * pz[1] + cc * pz[2];
                float* zz = &c->z[y * c->size + x];
                if (depth >= *zz) continue;
                if (alpha >= 1.0f) *zz = depth;
                int ix = c->x0 + x, iy = c->y0 + y;
                if (ix < 0 || iy < 0 || ix >= c->im->w || iy >= c->im->h) continue;
                unsigned char* p = c->im->rgb + ((size_t)iy * c->im->w + ix) * 3;
                float rgb[3] = { color.x * shade, color.y * shade, color.z * shade };
                for (int k = 0; k < 3; k++) p[k] = (unsigned char)fb_clamp(p[k] * (1 - alpha) + 255.0f * rgb[k] * alpha, 0, 255);
            }
    }
}

FbImage preview_sheet(const Design* d, const DesignMeshes* m, int cell) {
    const int cols = 5, rows = 2, top = 18;
    FbImage im = image_create(cols * cell, rows * cell + top, 236);
    const char* labels[5] = { "TARGET", "OPEN", "CLOSING", "CLOSING", "CLOSED" };
    for (int c = 0; c < cols; c++) image_text(&im, c * cell + 6, 5, 2, labels[c], 60, 60, 70);
    // (all in the design's space: the target brought into it through the placement)
    const float sc = fb_max(d->scale, 1e-4f);
    TriMesh tgt = mesh_copy(&d->target);
    for (int i = 0; i < tgt.nv; i++) tgt.pos[i] = v3scale(v3sub(tgt.pos[i], d->offset), 1.0f / sc);
    Box3 tb = mesh_bounds(&tgt);
    V3 tc = v3scale(v3add(tb.min, tb.max), 0.5f), te = v3sub(tb.max, tb.min);
    float half = 0.6f * fb_max(te.x, fb_max(te.y, te.z)) + 0.05f;
    const float halfClosed = 1.3f * d->R;
    Rigid* xf = FB_NEW(Rigid, FB_MAX_BODIES);
    float* z = FB_NEW(float, (size_t)cell * cell);
    const V3 dirs[2] = { v3norm(v3(0.35f, -0.3f, -1.0f)), v3norm(v3(-0.8f, -0.35f, 0.5f)) };
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            for (int k = 0; k < cell * cell; k++) z[k] = FLT_MAX;
            Canvas cv = { &im, z, c * cell, top + r * cell, cell, view_make(dirs[r], tc, half, cell) };
            if (c == 0) {
                draw_mesh(&cv, &tgt, NULL, v3(0.7f, 0.7f, 0.72f), 1.0f);
                continue;
            }
            float s = c == 1 ? 1.0f : (c == 2 ? 0.667f : (c == 3 ? 0.333f : 0.0f));
            design_transforms(d, s, xf);
            if (c > 1) {
                // framed around what's there (the motion can swing pieces outside both the open and closed frames)
                Box3 fb = box_empty();
                for (int b = 0; b < m->n && b < d->nbodies; b++)
                    for (int i = 0; i < m->bodies[b].nv; i += 7) fb = box_add(fb, rapply(xf[b], m->bodies[b].pos[i]));
                if (box_valid(fb)) {
                    V3 fe = v3sub(fb.max, fb.min);
                    cv.w = view_make(dirs[r], v3scale(v3add(fb.min, fb.max), 0.5f), fb_max(0.6f * fb_max(fe.x, fb_max(fe.y, fe.z)), halfClosed), cell);
                }
            }
            for (int b = 0; b < m->n && b < d->nbodies; b++) draw_mesh(&cv, &m->bodies[b], &xf[b], body_color(b), 1.0f);
        }
    mesh_free(&tgt);
    free(xf);
    free(z);
    return im;
}

// ---- a single tile (the test suite's video)

void preview_tile(FbImage* im, int x0, int y0, int size, const Design* d, const DesignMeshes* m, float s, float yaw, bool ghost) {
    // the background
    for (int y = y0; y < y0 + size && y < im->h; y++)
        for (int x = x0; x < x0 + size && x < im->w; x++) {
            unsigned char* p = im->rgb + ((size_t)y * im->w + x) * 3;
            float t = (float)(y - y0) / size;
            unsigned char g = (unsigned char)(34 + 14 * t);
            p[0] = g;
            p[1] = (unsigned char)(g + 2);
            p[2] = (unsigned char)(g + 8);
        }
    Rigid* xf = FB_NEW(Rigid, FB_MAX_BODIES);
    design_transforms(d, s, xf);
    // framed around the posed design and the target (both in the design's space)
    const float sc = fb_max(d->scale, 1e-4f);
    Box3 fb = box_empty();
    for (int b = 0; b < m->n && b < d->nbodies; b++)
        for (int i = 0; i < m->bodies[b].nv; i += 5) fb = box_add(fb, rapply(xf[b], m->bodies[b].pos[i]));
    TriMesh tgt = { 0 };
    if (ghost && d->target.nt) {
        tgt = mesh_copy(&d->target);
        for (int i = 0; i < tgt.nv; i++) {
            tgt.pos[i] = v3scale(v3sub(tgt.pos[i], d->offset), 1.0f / sc);
            fb = box_add(fb, tgt.pos[i]);
        }
    }
    if (!box_valid(fb)) fb = (Box3){ v3(-d->R, -d->R, -d->R), v3(d->R, d->R, d->R) };
    V3 e = v3sub(fb.max, fb.min);
    float half = 0.58f * fb_max(e.x, fb_max(e.y, e.z)) + 0.02f * d->R;
    V3 dir = v3norm(v3(0.35f * cosf(yaw) + sinf(yaw), -0.3f, -cosf(yaw) + 0.35f * sinf(yaw)));
    float* z = FB_NEW(float, (size_t)size * size);
    for (int k = 0; k < size * size; k++) z[k] = FLT_MAX;
    Canvas cv = { im, z, x0, y0, size, view_make(dir, v3scale(v3add(fb.min, fb.max), 0.5f), half, size) };
    for (int b = 0; b < m->n && b < d->nbodies; b++) draw_mesh(&cv, &m->bodies[b], &xf[b], body_color(b), 1.0f);
    // (the ghost fades out as it folds: closed, it's only the sphere)
    if (tgt.nt && s > 0.02f) draw_mesh(&cv, &tgt, NULL, v3(0.85f, 0.87f, 0.95f), 0.22f * fb_clamp(s, 0.0f, 1.0f));
    mesh_free(&tgt);
    free(z);
    free(xf);
}
