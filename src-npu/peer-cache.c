/* Peer-device weight cache for the Ascend NPU backend.
 *
 * hostbuf_file_reader_read() reloads every faulted model slice from the file
 * and copies it H2D. When several NPUs are visible the same slice can instead
 * be kept resident on a non-busy peer card; the next fault is then served by
 * a cross-device D2D copy (measured ~190 GB/s vs ~57 GB/s for pinned H2D).
 *
 * The pool is built lazily on the first cached read from device D: one LRU
 * list shared by every visible device except D, per-peer capacity accounting,
 * and a fill target picked by remaining headroom. A miss runs the normal
 * reader and then appends a D2D backfill on the same stream, so the copy is
 * ordered after the H2D that produced the data.
 */
#include "plat.h"
#include "acl-shim.h"
#include "peer-cache.h"
#include "thread-plat.h"

#define PEER_ALIGN             (2ULL * M)
#define PEER_MAX_ENTRY         (1ULL * G)
#define PEER_MIN_FREE          (1ULL * G)
#define PEER_DEFAULT_RATIO     0.8

bool hostbuf_file_reader_read(int device, uint64_t file_handle, uint64_t file_offset,
                              uint64_t size, cudaStream_t stream, uint64_t device_ptr,
                              bool mark_cold);

enum {
    PEER_CACHE_UNINIT = 0,
    PEER_CACHE_READY,
    PEER_CACHE_DISABLED,
};

typedef struct PeerCacheEntry {
    struct PeerCacheEntry *prev;
    struct PeerCacheEntry *next;
    uint64_t file_handle;
    uint64_t file_offset;
    uint64_t size;
    uint64_t alloc_size;
    int peer_slot;
    void *ptr;
    aclrtEvent ready;
} PeerCacheEntry;

static Mutex g_peer_lock;
static int g_peer_state = PEER_CACHE_UNINIT;
static bool g_peer_enabled;
static bool g_peer_env_checked;
static int g_peer_device = -1;
static int *g_peer_ids;
static uint64_t *g_peer_caps;
static uint64_t *g_peer_used;
static bool *g_peer_access;
static size_t g_peer_count;
static PeerCacheEntry *g_peer_lru_head;
static PeerCacheEntry *g_peer_lru_tail;
static uint64_t g_peer_hits;
static uint64_t g_peer_misses;
static uint64_t g_peer_fills;
static uint64_t g_peer_evictions;

static void peer_cache_ensure_lock(void) {
    if (!g_peer_lock) {
        g_peer_lock = mutex_create();
    }
}

SHARED_EXPORT
bool peer_cache_enabled(void) {
    const char *value;

    peer_cache_ensure_lock();
    if (g_peer_env_checked) {
        return g_peer_enabled;
    }
    value = getenv("AIMDO_NPU_PEER_CACHE");
    g_peer_enabled = !value || atoi(value) != 0;
    g_peer_env_checked = true;
    return g_peer_enabled;
}

static bool peer_cache_bind(int device) {
    return g_acl.p_aclrtSetDevice && g_acl.p_aclrtSetDevice((int32_t)device) == ACL_SUCCESS;
}

static uint64_t peer_cache_capacity(size_t free_bytes) {
    const char *gb = getenv("AIMDO_NPU_PEER_CACHE_GB");
    const char *ratio = getenv("AIMDO_NPU_PEER_CACHE_RATIO");
    double value;

    if (gb) {
        value = strtod(gb, NULL);
        if (value > 0.0) {
            return (uint64_t)(value * (double)G);
        }
    }
    value = ratio ? strtod(ratio, NULL) : PEER_DEFAULT_RATIO;
    if (!(value > 0.0) || value > 1.0) {
        value = PEER_DEFAULT_RATIO;
    }
    return (uint64_t)((double)free_bytes * value);
}

static void peer_cache_init(int device) {
    uint32_t device_count = 0;
    size_t peers = 0;

    mutex_lock(g_peer_lock);
    if (g_peer_state != PEER_CACHE_UNINIT) {
        mutex_unlock(g_peer_lock);
        return;
    }
    if (!peer_cache_enabled() ||
        !g_acl.p_aclrtGetDeviceCount || !g_acl.p_aclrtGetMemInfo ||
        !g_acl.p_aclrtMalloc || !g_acl.p_aclrtFree || !g_acl.p_aclrtSetDevice ||
        !g_acl.p_aclrtMemcpyAsync || !g_acl.p_aclrtCreateEvent ||
        !g_acl.p_aclrtRecordEvent || !g_acl.p_aclrtSynchronizeEvent ||
        !g_acl.p_aclrtDestroyEvent || !g_acl.p_aclrtDeviceCanAccessPeer ||
        !g_acl.p_aclrtDeviceEnablePeerAccess) {
        g_peer_state = PEER_CACHE_DISABLED;
        mutex_unlock(g_peer_lock);
        return;
    }
    if (g_acl.p_aclrtGetDeviceCount(&device_count) != ACL_SUCCESS || device_count < 2) {
        log(INFO, "comfy-aimdo NPU peer cache: fewer than two visible devices\n");
        g_peer_state = PEER_CACHE_DISABLED;
        mutex_unlock(g_peer_lock);
        return;
    }

    g_peer_ids = calloc(device_count, sizeof(*g_peer_ids));
    g_peer_caps = calloc(device_count, sizeof(*g_peer_caps));
    g_peer_used = calloc(device_count, sizeof(*g_peer_used));
    g_peer_access = calloc(device_count, sizeof(*g_peer_access));
    if (!g_peer_ids || !g_peer_caps || !g_peer_used || !g_peer_access) {
        free(g_peer_ids);
        free(g_peer_caps);
        free(g_peer_used);
        free(g_peer_access);
        g_peer_ids = NULL;
        g_peer_caps = NULL;
        g_peer_used = NULL;
        g_peer_access = NULL;
        g_peer_state = PEER_CACHE_DISABLED;
        mutex_unlock(g_peer_lock);
        return;
    }

    peer_cache_bind(device);
    for (uint32_t id = 0; id < device_count; id++) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        uint64_t capacity;
        int can_access = 0;

        if ((int)id == device) {
            continue;
        }
        if (!g_acl.p_aclrtDeviceCanAccessPeer ||
            g_acl.p_aclrtDeviceCanAccessPeer(&can_access, device, (int32_t)id) != ACL_SUCCESS ||
            !can_access ||
            g_acl.p_aclrtDeviceEnablePeerAccess((int32_t)id, 0) != ACL_SUCCESS) {
            log(WARNING, "comfy-aimdo NPU peer cache: device %u is not peer-accessible from %d\n",
                id, device);
            continue;
        }
        if (!peer_cache_bind((int)id) ||
            g_acl.p_aclrtGetMemInfo(ACL_HBM_MEM, &free_bytes, &total_bytes) != ACL_SUCCESS ||
            free_bytes < PEER_MIN_FREE) {
            log(WARNING, "comfy-aimdo NPU peer cache: device %u has no room (%zu MB free)\n",
                id, free_bytes / M);
            continue;
        }
        capacity = peer_cache_capacity(free_bytes);
        if (capacity > free_bytes) {
            capacity = free_bytes;
        }
        capacity &= ~(PEER_ALIGN - 1);
        if (capacity < PEER_ALIGN) {
            continue;
        }
        g_peer_ids[peers] = (int)id;
        g_peer_caps[peers] = capacity;
        g_peer_access[peers] = true;
        peers++;
        log(INFO, "comfy-aimdo NPU peer cache: peer device %u capacity %zu MB\n",
            id, (size_t)(capacity / M));
    }
    peer_cache_bind(device);

    if (!peers) {
        free(g_peer_ids);
        free(g_peer_caps);
        free(g_peer_used);
        free(g_peer_access);
        g_peer_ids = NULL;
        g_peer_caps = NULL;
        g_peer_used = NULL;
        g_peer_access = NULL;
        g_peer_state = PEER_CACHE_DISABLED;
        log(WARNING, "comfy-aimdo NPU peer cache disabled: no usable peer device\n");
        mutex_unlock(g_peer_lock);
        return;
    }

    g_peer_count = peers;
    g_peer_device = device;
    g_peer_state = PEER_CACHE_READY;
    log(INFO, "comfy-aimdo NPU peer cache enabled: %zu peer(s) for device %d\n", peers, device);
    mutex_unlock(g_peer_lock);
}

static void peer_cache_unlink(PeerCacheEntry *entry) {
    if (entry->prev) {
        entry->prev->next = entry->next;
    } else {
        g_peer_lru_head = entry->next;
    }
    if (entry->next) {
        entry->next->prev = entry->prev;
    } else {
        g_peer_lru_tail = entry->prev;
    }
}

static void peer_cache_push_front(PeerCacheEntry *entry) {
    entry->prev = NULL;
    entry->next = g_peer_lru_head;
    if (g_peer_lru_head) {
        g_peer_lru_head->prev = entry;
    } else {
        g_peer_lru_tail = entry;
    }
    g_peer_lru_head = entry;
}

static void peer_cache_free_peer(int peer_slot, void *ptr, int restore_device) {
    aclError err;

    if (!peer_cache_bind(g_peer_ids[peer_slot])) {
        log(WARNING, "comfy-aimdo NPU peer cache: cannot bind device %d to free entry\n",
            g_peer_ids[peer_slot]);
        return;
    }
    err = g_acl.p_aclrtFree(ptr);
    if (err != ACL_SUCCESS) {
        log(WARNING, "comfy-aimdo NPU peer cache: aclrtFree on device %d failed (%d)\n",
            g_peer_ids[peer_slot], (int)err);
    }
    peer_cache_bind(restore_device);
}

static void peer_cache_entry_destroy(PeerCacheEntry *entry, int restore_device) {
    if (entry->ready) {
        g_acl.p_aclrtSynchronizeEvent(entry->ready);
        g_acl.p_aclrtDestroyEvent(entry->ready);
    }
    peer_cache_free_peer(entry->peer_slot, entry->ptr, restore_device);
    g_peer_used[entry->peer_slot] -= entry->alloc_size;
    free(entry);
    g_peer_evictions++;
}

static bool peer_cache_make_room(int peer_slot, uint64_t size, int restore_device) {
    while (g_peer_used[peer_slot] + size > g_peer_caps[peer_slot]) {
        PeerCacheEntry *victim = g_peer_lru_tail;

        while (victim && victim->peer_slot != peer_slot) {
            victim = victim->prev;
        }
        if (!victim) {
            return false;
        }
        peer_cache_unlink(victim);
        peer_cache_entry_destroy(victim, restore_device);
    }
    return true;
}

static void peer_cache_fill(int device, uint64_t file_handle, uint64_t file_offset,
                            uint64_t size, CUstream stream, uint64_t device_ptr) {
    uint64_t alloc_size = ALIGN_UP(size, PEER_ALIGN);
    PeerCacheEntry *entry;
    int best_slot = -1;
    uint64_t best_room = 0;
    void *ptr = NULL;

    mutex_lock(g_peer_lock);
    if (g_peer_state != PEER_CACHE_READY) {
        mutex_unlock(g_peer_lock);
        return;
    }
    for (entry = g_peer_lru_head; entry; entry = entry->next) {
        if (entry->file_handle == file_handle && entry->file_offset == file_offset &&
            entry->size == size) {
            mutex_unlock(g_peer_lock);
            return;
        }
    }
    for (size_t i = 0; i < g_peer_count; i++) {
        uint64_t room = g_peer_caps[i] > g_peer_used[i] ? g_peer_caps[i] - g_peer_used[i] : 0;

        if (room > best_room || (best_slot < 0 && g_peer_caps[i] >= alloc_size)) {
            best_room = room;
            best_slot = (int)i;
        }
    }
    if (best_slot < 0 ||
        !peer_cache_make_room(best_slot, alloc_size, device) ||
        !peer_cache_bind(g_peer_ids[best_slot]) ||
        g_acl.p_aclrtMalloc(&ptr, (size_t)alloc_size, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        !ptr) {
        if (ptr) {
            g_acl.p_aclrtFree(ptr);
        }
        peer_cache_bind(device);
        mutex_unlock(g_peer_lock);
        return;
    }
    peer_cache_bind(device);

    entry = calloc(1, sizeof(*entry));
    if (!entry) {
        peer_cache_free_peer(best_slot, ptr, device);
        mutex_unlock(g_peer_lock);
        return;
    }
    entry->file_handle = file_handle;
    entry->file_offset = file_offset;
    entry->size = size;
    entry->alloc_size = alloc_size;
    entry->peer_slot = best_slot;
    entry->ptr = ptr;

    if (g_acl.p_aclrtMemcpyAsync(ptr, (size_t)alloc_size, (void *)(uintptr_t)device_ptr,
                                 (size_t)size, ACL_MEMCPY_DEVICE_TO_DEVICE,
                                 (aclrtStream)stream) != ACL_SUCCESS) {
        peer_cache_free_peer(best_slot, ptr, device);
        free(entry);
        mutex_unlock(g_peer_lock);
        return;
    }
    if (g_acl.p_aclrtCreateEvent(&entry->ready) == ACL_SUCCESS) {
        if (g_acl.p_aclrtRecordEvent(entry->ready, (aclrtStream)stream) != ACL_SUCCESS) {
            g_acl.p_aclrtDestroyEvent(entry->ready);
            entry->ready = NULL;
        }
    } else {
        entry->ready = NULL;
    }

    g_peer_used[best_slot] += alloc_size;
    peer_cache_push_front(entry);
    g_peer_fills++;
    mutex_unlock(g_peer_lock);
}

SHARED_EXPORT
bool peer_cache_read_file_to_device(uint64_t file_handle, uint64_t file_offset, uint64_t size,
                                    CUstream stream, uint64_t device_ptr, int device,
                                    bool mark_cold) {
    PeerCacheEntry *entry;
    bool hit = false;

    if (size == 0 || size > PEER_MAX_ENTRY || !device_ptr || device < 0 ||
        !peer_cache_enabled()) {
        return hostbuf_file_reader_read(device, file_handle, file_offset, size,
                                        (cudaStream_t)stream, device_ptr, mark_cold);
    }

    peer_cache_init(device);
    if (g_peer_state != PEER_CACHE_READY || device != g_peer_device) {
        return hostbuf_file_reader_read(device, file_handle, file_offset, size,
                                        (cudaStream_t)stream, device_ptr, mark_cold);
    }

    mutex_lock(g_peer_lock);
    for (entry = g_peer_lru_head; entry; entry = entry->next) {
        if (entry->file_handle != file_handle || entry->file_offset != file_offset ||
            entry->size != size) {
            continue;
        }
        /* Wait for the fill copy before handing the peer buffer to the stream. */
        if (entry->ready) {
            g_acl.p_aclrtSynchronizeEvent(entry->ready);
        }
        peer_cache_unlink(entry);
        peer_cache_push_front(entry);
        if (g_acl.p_aclrtMemcpyAsync((void *)(uintptr_t)device_ptr, (size_t)size, entry->ptr,
                                     (size_t)size, ACL_MEMCPY_DEVICE_TO_DEVICE,
                                     (aclrtStream)stream) == ACL_SUCCESS) {
            g_peer_hits++;
            hit = true;
        } else {
            log(WARNING, "comfy-aimdo NPU peer cache: D2D restore failed, dropping entry\n");
            peer_cache_unlink(entry);
            peer_cache_entry_destroy(entry, device);
        }
        break;
    }
    if (!hit) {
        g_peer_misses++;
    }
    mutex_unlock(g_peer_lock);

    if (hit) {
        return true;
    }
    if (!hostbuf_file_reader_read(device, file_handle, file_offset, size, (cudaStream_t)stream,
                                  device_ptr, mark_cold)) {
        return false;
    }
    peer_cache_fill(device, file_handle, file_offset, size, stream, device_ptr);
    return true;
}

SHARED_EXPORT
void peer_cache_stats(uint64_t *hits, uint64_t *misses, uint64_t *fills, uint64_t *evictions) {
    peer_cache_ensure_lock();
    mutex_lock(g_peer_lock);
    if (hits) {
        *hits = g_peer_hits;
    }
    if (misses) {
        *misses = g_peer_misses;
    }
    if (fills) {
        *fills = g_peer_fills;
    }
    if (evictions) {
        *evictions = g_peer_evictions;
    }
    mutex_unlock(g_peer_lock);
}

SHARED_EXPORT
void peer_cache_cleanup(void) {
    int current_device = -1;

    if (!g_peer_lock) {
        return;
    }

    mutex_lock(g_peer_lock);
    if (g_acl.p_aclrtGetDevice) {
        g_acl.p_aclrtGetDevice(&current_device);
    }
    /* Drain in-flight cross-device fills before releasing the peer buffers. */
    if (g_peer_device >= 0 && peer_cache_bind(g_peer_device) &&
        g_acl.p_aclrtSynchronizeDevice) {
        g_acl.p_aclrtSynchronizeDevice();
    }
    /* Drop peer mappings before freeing: active mappings pin the pages. The
     * access was enabled from the owner's context, so disable from there too. */
    if (g_acl.p_aclrtDeviceDisablePeerAccess && g_peer_access && g_peer_device >= 0 &&
        peer_cache_bind(g_peer_device)) {
        for (size_t i = 0; i < g_peer_count; i++) {
            if (g_peer_access[i] &&
                g_acl.p_aclrtDeviceDisablePeerAccess(g_peer_ids[i]) != ACL_SUCCESS) {
                log(WARNING, "comfy-aimdo NPU peer cache: cannot disable peer access on device %d\n",
                    g_peer_ids[i]);
            }
        }
    }
    while (g_peer_lru_head) {
        PeerCacheEntry *entry = g_peer_lru_head;

        peer_cache_unlink(entry);
        if (entry->ready) {
            g_acl.p_aclrtSynchronizeEvent(entry->ready);
            g_acl.p_aclrtDestroyEvent(entry->ready);
        }
        peer_cache_free_peer(entry->peer_slot, entry->ptr, g_peer_device);
        g_peer_used[entry->peer_slot] -= entry->alloc_size;
        free(entry);
        g_peer_evictions++;
    }
    if (current_device >= 0) {
        peer_cache_bind(current_device);
    }
    free(g_peer_ids);
    free(g_peer_caps);
    free(g_peer_used);
    free(g_peer_access);
    g_peer_ids = NULL;
    g_peer_caps = NULL;
    g_peer_used = NULL;
    g_peer_access = NULL;
    g_peer_count = 0;
    g_peer_device = -1;
    g_peer_state = PEER_CACHE_UNINIT;
    mutex_unlock(g_peer_lock);
}
