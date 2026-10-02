/* Peer-device weight cache for the Ascend NPU backend.
 *
 * When several NPUs are visible, model slices faulted back in by
 * hostbuf_file_reader_read() can be kept resident on a non-busy peer card and
 * restored with a cross-device D2D copy instead of a file read plus H2D.
 * Only built into aimdo_npu.so.
 */
#pragma once

#include "gpu_abi.h"

#include <stdbool.h>
#include <stdint.h>

/* True when AIMDO_NPU_PEER_CACHE did not disable the feature. This only
 * reports the configured intent; the first cached read still lazily builds
 * the peer pool and degrades to the normal reader if that fails. */
bool peer_cache_enabled(void);

/* Drop-in replacement for hostbuf_file_reader_read() used by the Python
 * host_buffer bridge. Falls back to the normal reader whenever the cache is
 * disabled, the entry is too large, or the peer pool is unavailable. */
bool peer_cache_read_file_to_device(uint64_t file_handle, uint64_t file_offset,
                                    uint64_t size, CUstream stream, uint64_t device_ptr,
                                    int device, bool mark_cold);

/* Test and observability counters; any output pointer may be NULL. */
void peer_cache_stats(uint64_t *hits, uint64_t *misses, uint64_t *fills,
                      uint64_t *evictions);

/* Frees every cached peer allocation. Called from control.c's cleanup(). */
void peer_cache_cleanup(void);
