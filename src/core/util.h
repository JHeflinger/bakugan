#ifndef FB_UTIL_H
#define FB_UTIL_H
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>

// ---- memory: checked allocation (aborts with a message when out of memory)
void* fb_alloc(size_t bytes);              // zeroed
void* fb_realloc(void* p, size_t bytes);
char* fb_strdup(const char* s);
#define FB_NEW(T, n) ((T*)fb_alloc(sizeof(T) * (size_t)(n)))

// ---- a growable array of anything: FB_ARRAY(T) name; then fb_push(&name, value)
#define FB_ARRAY(T) struct { T* data; int count, cap; }
#define fb_push(arr, value) do { \
        if ((arr)->count >= (arr)->cap) { \
            (arr)->cap = (arr)->cap ? (arr)->cap * 2 : 16; \
            (arr)->data = fb_realloc((arr)->data, sizeof(*(arr)->data) * (size_t)(arr)->cap); \
        } \
        (arr)->data[(arr)->count++] = (value); \
    } while (0)
#define fb_free_array(arr) do { free((arr)->data); (arr)->data = NULL; (arr)->count = (arr)->cap = 0; } while (0)

// ---- time
double fb_now(void);                       // seconds, monotonic
void fb_sleep_ms(int ms);

// ---- text
char* fb_format(const char* fmt, ...);     // malloc'd
char* fb_read_file(const char* path, size_t* size);
bool fb_write_file(const char* path, const void* data, size_t size);
bool fb_make_dir(const char* path);
bool fb_file_exists(const char* path);
// The files in dir ending in ext (case ignored), sorted, as malloc'd names (and the array); -1 if dir can't be read.
int fb_list_dir(const char* dir, const char* ext, char*** out);
const char* fb_basename(const char* path); // pointer into path
char* fb_stem(const char* path);           // malloc'd file name without folder and extension

// ---- threads: a persistent pool. fb_parallel_for runs fn(ctx, i) for i in [0, n) on every worker; called from
// inside a worker it runs serially (no nesting).
typedef void (*FbTask)(void* ctx, int i);
void fb_threads_init(int threads);         // 0: every core
int fb_threads_count(void);
void fb_parallel_for(int n, FbTask fn, void* ctx);
void fb_threads_shutdown(void);

typedef struct FbMutex FbMutex;
FbMutex* fb_mutex_create(void);
void fb_mutex_lock(FbMutex* m);
void fb_mutex_unlock(FbMutex* m);
void fb_mutex_destroy(FbMutex* m);

typedef struct FbThread FbThread;
FbThread* fb_thread_start(void (*fn)(void*), void* arg);
void fb_thread_join(FbThread* t);
#endif
