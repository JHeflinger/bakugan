#include "util.h"
#include <pthread.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#else
#include <unistd.h>
#endif

void* fb_alloc(size_t bytes) {
    void* p = calloc(1, bytes ? bytes : 1);
    if (!p) { fprintf(stderr, "foldball: out of memory (%zu bytes)\n", bytes); abort(); }
    return p;
}

void* fb_realloc(void* p, size_t bytes) {
    void* q = realloc(p, bytes ? bytes : 1);
    if (!q) { fprintf(stderr, "foldball: out of memory (%zu bytes)\n", bytes); abort(); }
    return q;
}

char* fb_strdup(const char* s) {
    size_t n = strlen(s ? s : "");
    char* r = fb_alloc(n + 1);
    memcpy(r, s ? s : "", n);
    return r;
}

double fb_now(void) {
#ifdef _WIN32
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

void fb_sleep_ms(int ms) {
#ifdef _WIN32
    Sleep(ms);
#else
    struct timespec ts = { ms / 1000, (long)(ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
#endif
}

char* fb_format(const char* fmt, ...) {
    va_list a, b;
    va_start(a, fmt);
    va_copy(b, a);
    int n = vsnprintf(NULL, 0, fmt, a);
    va_end(a);
    char* s = fb_alloc((size_t)n + 1);
    vsnprintf(s, (size_t)n + 1, fmt, b);
    va_end(b);
    return s;
}

char* fb_read_file(const char* path, size_t* size) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); return NULL; }
    char* d = fb_alloc((size_t)n + 1);
    size_t got = fread(d, 1, (size_t)n, f);
    fclose(f);
    d[got] = 0;
    if (size) *size = got;
    return d;
}

bool fb_write_file(const char* path, const void* data, size_t size) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    bool ok = fwrite(data, 1, size, f) == size;
    return fclose(f) == 0 && ok;
}

bool fb_make_dir(const char* path) {
    // every folder along the way
    char* p = fb_strdup(path);
    for (char* c = p + 1; *c; c++) {
        if (*c != '/' && *c != '\\') continue;
        char keep = *c;
        *c = 0;
#ifdef _WIN32
        _mkdir(p);
#else
        mkdir(p, 0755);
#endif
        *c = keep;
    }
#ifdef _WIN32
    _mkdir(p);
#else
    mkdir(p, 0755);
#endif
    struct stat st;
    bool ok = stat(p, &st) == 0 && (st.st_mode & S_IFDIR);
    free(p);
    return ok;
}

#ifdef _WIN32
#include <io.h>
#define strcasecmp _stricmp
#else
#include <dirent.h>
#include <strings.h>
#endif
static int cmp_str(const void* a, const void* b) { return strcmp(*(char* const*)a, *(char* const*)b); }

int fb_list_dir(const char* dir, const char* ext, char*** out) {
    *out = NULL;
    int n = 0, cap = 0;
    char** names = NULL;
    const size_t el = strlen(ext);
#define FB_ADD(nm) do { \
        const char* nm_ = (nm); size_t l_ = strlen(nm_); \
        if (l_ > el && strcasecmp(nm_ + l_ - el, ext) == 0) { \
            if (n >= cap) { cap = cap ? 2 * cap : 16; names = fb_realloc(names, sizeof(char*) * (size_t)cap); } \
            names[n++] = fb_strdup(nm_); } } while (0)
#ifdef _WIN32
    char* pattern = fb_format("%s\\*", dir);
    struct _finddata_t fd;
    intptr_t h = _findfirst(pattern, &fd);
    free(pattern);
    if (h == -1) return -1;
    do { if (!(fd.attrib & _A_SUBDIR)) FB_ADD(fd.name); } while (_findnext(h, &fd) == 0);
    _findclose(h);
#else
    DIR* d = opendir(dir);
    if (!d) return -1;
    struct dirent* e;
    while ((e = readdir(d)) != NULL) FB_ADD(e->d_name);
    closedir(d);
#endif
#undef FB_ADD
    if (n > 1) qsort(names, (size_t)n, sizeof(char*), cmp_str);
    *out = names;
    return n;
}

bool fb_file_exists(const char* path) {
    struct stat st;
    return stat(path, &st) == 0;
}

const char* fb_basename(const char* path) {
    const char* b = path;
    for (const char* c = path; *c; c++)
        if (*c == '/' || *c == '\\') b = c + 1;
    return b;
}

char* fb_stem(const char* path) {
    char* s = fb_strdup(fb_basename(path));
    char* dot = strrchr(s, '.');
    if (dot && dot != s) *dot = 0;
    return s;
}

// ---- the thread pool

static struct {
    int workers;                 // threads besides the caller
    pthread_t* threads;
    pthread_mutex_t lock;
    pthread_cond_t wake, done;
    FbTask fn;
    void* ctx;
    int n, next, active, generation;
    bool quit, ready;
} P;

static __thread bool tInWorker = false;

static void run_items(void) {
    for (;;) {
        pthread_mutex_lock(&P.lock);
        int i = P.next < P.n ? P.next++ : -1;
        pthread_mutex_unlock(&P.lock);
        if (i < 0) return;
        P.fn(P.ctx, i);
    }
}

static void* worker_main(void* arg) {
    (void)arg;
    tInWorker = true;
    int seen = 0;
    pthread_mutex_lock(&P.lock);
    for (;;) {
        while (!P.quit && P.generation == seen) pthread_cond_wait(&P.wake, &P.lock);
        if (P.quit) break;
        seen = P.generation;
        P.active++;
        pthread_mutex_unlock(&P.lock);
        run_items();
        pthread_mutex_lock(&P.lock);
        if (--P.active == 0) pthread_cond_broadcast(&P.done);
    }
    pthread_mutex_unlock(&P.lock);
    return NULL;
}

static int core_count(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
#endif
}

void fb_threads_init(int threads) {
    if (P.ready) return;
    int n = threads > 0 ? threads : core_count();
    P.workers = n - 1;
    pthread_mutex_init(&P.lock, NULL);
    pthread_cond_init(&P.wake, NULL);
    pthread_cond_init(&P.done, NULL);
    P.threads = FB_NEW(pthread_t, P.workers > 0 ? P.workers : 1);
    for (int k = 0; k < P.workers; k++) pthread_create(&P.threads[k], NULL, worker_main, NULL);
    P.ready = true;
}

int fb_threads_count(void) { return P.ready ? P.workers + 1 : 1; }

// One parallel loop at a time: callers from different threads (the optimizer, the GUI's mesher) take turns.
static pthread_mutex_t gLoop = PTHREAD_MUTEX_INITIALIZER;

void fb_parallel_for(int n, FbTask fn, void* ctx) {
    if (n <= 0) return;
    if (!P.ready || P.workers == 0 || tInWorker || n == 1) {
        for (int i = 0; i < n; i++) fn(ctx, i);
        return;
    }
    pthread_mutex_lock(&gLoop);
    pthread_mutex_lock(&P.lock);
    P.fn = fn;
    P.ctx = ctx;
    P.n = n;
    P.next = 0;
    P.generation++;
    pthread_cond_broadcast(&P.wake);
    pthread_mutex_unlock(&P.lock);
    bool was = tInWorker;
    tInWorker = true;            // (nested loops inside fn run serially)
    run_items();
    tInWorker = was;
    pthread_mutex_lock(&P.lock);
    while (P.active > 0 || P.next < P.n) pthread_cond_wait(&P.done, &P.lock);
    pthread_mutex_unlock(&P.lock);
    pthread_mutex_unlock(&gLoop);
}

void fb_threads_shutdown(void) {
    if (!P.ready) return;
    pthread_mutex_lock(&P.lock);
    P.quit = true;
    pthread_cond_broadcast(&P.wake);
    pthread_mutex_unlock(&P.lock);
    for (int k = 0; k < P.workers; k++) pthread_join(P.threads[k], NULL);
    free(P.threads);
    P.ready = false;
}

struct FbMutex { pthread_mutex_t m; };
FbMutex* fb_mutex_create(void) {
    FbMutex* m = FB_NEW(FbMutex, 1);
    pthread_mutex_init(&m->m, NULL);
    return m;
}
void fb_mutex_lock(FbMutex* m) { pthread_mutex_lock(&m->m); }
void fb_mutex_unlock(FbMutex* m) { pthread_mutex_unlock(&m->m); }
void fb_mutex_destroy(FbMutex* m) {
    if (!m) return;
    pthread_mutex_destroy(&m->m);
    free(m);
}

struct FbThread { pthread_t t; void (*fn)(void*); void* arg; };
static void* thread_tramp(void* a) {
    FbThread* t = a;
    t->fn(t->arg);
    return NULL;
}
FbThread* fb_thread_start(void (*fn)(void*), void* arg) {
    FbThread* t = FB_NEW(FbThread, 1);
    t->fn = fn;
    t->arg = arg;
    if (pthread_create(&t->t, NULL, thread_tramp, t) != 0) { free(t); return NULL; }
    return t;
}
void fb_thread_join(FbThread* t) {
    if (!t) return;
    pthread_join(t->t, NULL);
    free(t);
}
