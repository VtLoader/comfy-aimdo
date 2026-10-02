/* Ascend NPU adaptation of the shared AimdoCudaDispatch contract.
 *
 * Each npu_cu* entry point translates the CUDA driver call shape used by the
 * portable core into the corresponding CANN aclrt* call. The raw ACL function
 * pointers live in g_acl and are populated by src-npu/dispatch.c.
 */
#include "plat.h"
#include "acl-shim.h"
#include "thread-plat.h"

#define CUDA_ERROR_INVALID_VALUE 1
#define CUDA_ERROR_OUT_OF_MEMORY 2
#define CUDA_ERROR_NOT_SUPPORTED 801
#define CUDA_ERROR_UNKNOWN       999

AimdoAclDispatch g_acl;

/* Ranges successfully passed to aclrtHostRegister, guarded by a mutex.
 * aclrtHostUnregister faults or errors for pointers that were never
 * registered, so the shim consults this table first. Entries are added by
 * both the shim adapter and the raw-ACL funchook, and removed on successful
 * unregistration. */
typedef struct {
    void *ptr;
    size_t size;
} NpuHostRegRange;

static Mutex g_hostreg_mutex;
static NpuHostRegRange *g_hostreg_ranges;
static size_t g_hostreg_count;
static size_t g_hostreg_capacity;

void npu_hostreg_init(void) {
    if (!g_hostreg_mutex) {
        g_hostreg_mutex = mutex_create();
    }
}

void npu_hostreg_cleanup(void) {
    mutex_destroy(g_hostreg_mutex);
    g_hostreg_mutex = NULL;
    free(g_hostreg_ranges);
    g_hostreg_ranges = NULL;
    g_hostreg_count = 0;
    g_hostreg_capacity = 0;
}

void npu_hostreg_add(void *ptr, size_t size) {
    if (!g_hostreg_mutex || !size) {
        return;
    }

    mutex_lock(g_hostreg_mutex);
    for (size_t i = 0; i < g_hostreg_count; i++) {
        if (g_hostreg_ranges[i].ptr == ptr) {
            g_hostreg_ranges[i].size = size;
            goto out;
        }
    }
    if (g_hostreg_count == g_hostreg_capacity) {
        size_t capacity = g_hostreg_capacity ? g_hostreg_capacity * 2 : 16;
        NpuHostRegRange *ranges = realloc(g_hostreg_ranges, capacity * sizeof(*ranges));

        if (!ranges) {
            goto out;
        }
        g_hostreg_ranges = ranges;
        g_hostreg_capacity = capacity;
    }
    g_hostreg_ranges[g_hostreg_count].ptr = ptr;
    g_hostreg_ranges[g_hostreg_count].size = size;
    g_hostreg_count++;

out:
    mutex_unlock(g_hostreg_mutex);
}

void npu_hostreg_remove(void *ptr) {
    if (!g_hostreg_mutex) {
        return;
    }

    mutex_lock(g_hostreg_mutex);
    for (size_t i = 0; i < g_hostreg_count; i++) {
        if (g_hostreg_ranges[i].ptr == ptr) {
            g_hostreg_ranges[i] = g_hostreg_ranges[--g_hostreg_count];
            break;
        }
    }
    mutex_unlock(g_hostreg_mutex);
}

bool npu_hostreg_contains(void *ptr) {
    bool found = false;

    if (!g_hostreg_mutex) {
        return false;
    }

    mutex_lock(g_hostreg_mutex);
    for (size_t i = 0; i < g_hostreg_count; i++) {
        if (g_hostreg_ranges[i].ptr == ptr) {
            found = true;
            break;
        }
    }
    mutex_unlock(g_hostreg_mutex);
    return found;
}

static CUresult npu_map_error(aclError err) {
    switch (err) {
    case ACL_SUCCESS:
        return CUDA_SUCCESS;
    case ACL_ERROR_INVALID_PARAM:          /* 100000 */
    case ACL_ERROR_RT_PARAM_INVALID:       /* 107000, e.g. bad alignment/memAttr */
        return CUDA_ERROR_INVALID_VALUE;
    case ACL_ERROR_RT_FEATURE_NOT_SUPPORT: /* 207000 */
        return CUDA_ERROR_NOT_SUPPORTED;
    case ACL_ERROR_RT_MEMORY_ALLOCATION:   /* 207001 */
        return CUDA_ERROR_OUT_OF_MEMORY;
    case ACL_ERROR_RT_TS_ERROR:            /* 507001, async copy/sync failure */
    case ACL_ERROR_RT_DRV_INTERNAL_ERROR:  /* 507899, bad unmap/overlap */
    default:
        return CUDA_ERROR_UNKNOWN;
    }
}

static CUresult npu_hal_map_error(int err) {
    switch (err) {
    case DRV_ERROR_NONE:
        return CUDA_SUCCESS;
    case DRV_ERROR_INVALID_VALUE:
    case DRV_ERROR_PARA_ERROR:
        return CUDA_ERROR_INVALID_VALUE;
    case DRV_ERROR_OUT_OF_MEMORY:
    case DRV_ERROR_NO_RESOURCES:
        return CUDA_ERROR_OUT_OF_MEMORY;
    default:
        return CUDA_ERROR_UNKNOWN;
    }
}

/* CUDA events only need to order the async H2D copies started by the
 * portable core; aclrtSynchronizeEvent alone does not surface a failed
 * aclrtMemcpyAsync. Remember the recording stream so event waits can also
 * synchronize it, which is where CANN reports the copy failure (507001). */
typedef struct {
    aclrtEvent event;
    aclrtStream stream;
} NpuEvent;

CUresult CUDAAPI npu_cuInit(unsigned int flags) {
    uint32_t count = 0;

    (void)flags;
    return npu_map_error(g_acl.p_aclrtGetDeviceCount(&count));
}

CUresult CUDAAPI npu_cuGetErrorString(CUresult error, const char **pStr) {
    const char *message = g_acl.p_aclGetRecentErrMsg ? g_acl.p_aclGetRecentErrMsg() : NULL;

    (void)error;
    if (pStr) {
        *pStr = message ? message : "ACL runtime error";
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI npu_cuCtxGetDevice(CUdevice *device) {
    int32_t device_id = 0;
    aclError err = g_acl.p_aclrtGetDevice(&device_id);

    if (err == ACL_SUCCESS) {
        *device = (CUdevice)device_id;
    }
    return npu_map_error(err);
}

CUresult CUDAAPI npu_cuCtxSynchronize(void) {
    return npu_map_error(g_acl.p_aclrtSynchronizeDevice());
}

CUresult CUDAAPI npu_cuDeviceGet(CUdevice *device, int ordinal) {
    *device = (CUdevice)ordinal;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI npu_cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
    (void)dev;
    if (attrib == CU_DEVICE_ATTRIBUTE_INTEGRATED) {
        *pi = 0; /* Ascend 910A is a discrete HBM device. */
        return CUDA_SUCCESS;
    }
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult CUDAAPI npu_cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    aclError err;

    if (!g_acl.p_aclrtSetDevice) {
        return CUDA_ERROR_UNKNOWN;
    }
    /* aclrtGetMemInfo reads the memory of the context bound to the calling
     * thread. aclrtSetDevice creates and binds the primary context for the
     * requested device, which a bare process (no torch_npu) does not have. */
    err = g_acl.p_aclrtSetDevice((int32_t)dev);
    if (err != ACL_SUCCESS) {
        return npu_map_error(err);
    }
    err = g_acl.p_aclrtGetMemInfo(ACL_HBM_MEM, &free_bytes, &total_bytes);
    if (err != ACL_SUCCESS) {
        return npu_map_error(err);
    }
    *bytes = total_bytes;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI npu_cuDeviceGetName(char *name, int len, CUdevice dev) {
    const char *soc_name;

    (void)dev;
    soc_name = g_acl.p_aclrtGetSocName();
    snprintf(name, (size_t)len, "%s", soc_name ? soc_name : "Ascend NPU");
    return CUDA_SUCCESS;
}

CUresult CUDAAPI npu_cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
    if (!g_acl.p_aclrtDeviceGetUuid) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return npu_map_error(g_acl.p_aclrtDeviceGetUuid((int32_t)dev, uuid));
}

CUresult CUDAAPI npu_cuMemGetInfo(size_t *free_bytes, size_t *total_bytes) {
    return npu_map_error(g_acl.p_aclrtGetMemInfo(ACL_HBM_MEM, free_bytes, total_bytes));
}

CUresult CUDAAPI npu_cuMemAlloc(CUdeviceptr *dptr, size_t size) {
    return npu_map_error(g_acl.p_aclrtMalloc((void **)dptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
}

CUresult CUDAAPI npu_cuMemFree(CUdeviceptr dptr) {
    return npu_map_error(g_acl.p_aclrtFree((void *)(uintptr_t)dptr));
}

CUresult CUDAAPI npu_cuMemAllocHost(void **pp, size_t bytesize) {
    return npu_map_error(g_acl.p_aclrtMallocHost(pp, bytesize));
}

CUresult CUDAAPI npu_cuMemFreeHost(void *p) {
    return npu_map_error(g_acl.p_aclrtFreeHost(p));
}

CUresult CUDAAPI npu_cuMemHostRegister(void *p, size_t bytesize, unsigned int flags) {
    CUresult result;

    (void)flags; /* The core always passes 0; hostbuf memory is malloc'd, not aclrtMallocHost'd. */
    if (!g_acl.p_aclrtHostRegister) {
        return CUDA_ERROR_UNKNOWN;
    }
    /* The CANN host register interface (dlsym symbol in dispatch.c) requires
     * PINNED semantics so aclrtMemcpyAsync can consume the range; the device
     * pointer it exposes is not used for H2D. */
    result = npu_map_error(g_acl.p_aclrtHostRegister(p, bytesize, ACL_HOST_REG_PINNED));
    if (result == CUDA_SUCCESS) {
        npu_hostreg_add(p, bytesize);
    }
    return result;
}

CUresult CUDAAPI npu_cuMemHostUnregister(void *p) {
    CUresult result;

    if (!g_acl.p_aclrtHostUnregister || !npu_hostreg_contains(p)) {
        return CUDA_SUCCESS;
    }
    result = npu_map_error(g_acl.p_aclrtHostUnregister(p));
    if (result == CUDA_SUCCESS) {
        npu_hostreg_remove(p);
    }
    return result;
}

CUresult CUDAAPI npu_cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment,
                                         CUdeviceptr addr, unsigned long long flags) {
    void *vir_ptr = NULL;
    aclError err;

    (void)alignment; /* Both paths only accept alignment 0. */
    (void)addr;      /* expectPtr must be NULL. */
    (void)flags;

    /* The HAL allocates arbitrary-size VA (up to its arena limit), while
     * aclrtReserveMemAddress rejects a single reservation >= 2 GiB. */
    if (g_acl.hal_mem_address_available) {
        int ret = g_acl.p_halMemAddressReserve(&vir_ptr, size, 0, NULL, 0);

        if (ret == DRV_ERROR_NONE) {
            *ptr = (CUdeviceptr)(uintptr_t)vir_ptr;
        }
        return npu_hal_map_error(ret);
    }

    if (size > (size_t)INT32_MAX) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    err = g_acl.p_aclrtReserveMemAddress(&vir_ptr, size, 0, NULL, 0);
    if (err == ACL_SUCCESS) {
        *ptr = (CUdeviceptr)(uintptr_t)vir_ptr;
    }
    return npu_map_error(err);
}

CUresult CUDAAPI npu_cuMemAddressFree(CUdeviceptr ptr, size_t size) {
    (void)size;
    if (g_acl.hal_mem_address_available) {
        return npu_hal_map_error(g_acl.p_halMemAddressFree((void *)(uintptr_t)ptr));
    }
    return npu_map_error(g_acl.p_aclrtReleaseMemAddress((void *)(uintptr_t)ptr));
}

CUresult CUDAAPI npu_cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                                 const CUmemAllocationProp *prop, unsigned long long flags) {
    /* Physical allocations are 2 MiB-granular; the core maps them in 16 MiB
     * and 32 MiB chunks. HBM_MEM_HUGE is the only accepted memAttr here. */
    AclPhysicalMemProp acl_prop = {
        .handleType = ACL_MEM_HANDLE_TYPE_NONE,
        .allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED,
        .memAttr = ACL_HBM_MEM_HUGE,
        .location = {
            .id = (uint32_t)prop->location.id,
            .type = ACL_MEM_LOCATION_TYPE_DEVICE,
        },
        .reserve = 0,
    };
    aclrtDrvMemHandle acl_handle = NULL;
    aclError err = g_acl.p_aclrtMallocPhysical(&acl_handle, size, &acl_prop, (uint64_t)flags);

    if (err == ACL_SUCCESS) {
        *handle = (CUmemGenericAllocationHandle)(uintptr_t)acl_handle;
    }
    return npu_map_error(err);
}

CUresult CUDAAPI npu_cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                              CUmemGenericAllocationHandle handle, unsigned long long flags) {
    return npu_map_error(g_acl.p_aclrtMapMem((void *)(uintptr_t)ptr, size, offset,
                                             (aclrtDrvMemHandle)(uintptr_t)handle,
                                             (uint64_t)flags));
}

CUresult CUDAAPI npu_cuMemSetAccess(CUdeviceptr ptr, size_t size,
                                    const CUmemAccessDesc *desc, size_t count) {
    AclMemAccessDesc *acl_desc;
    CUresult result;

    if (!count) {
        return CUDA_SUCCESS;
    }
    acl_desc = calloc(count, sizeof(*acl_desc));
    if (!acl_desc) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    for (size_t i = 0; i < count; i++) {
        acl_desc[i].flags = (int)desc[i].flags;
        acl_desc[i].location.id = (uint32_t)desc[i].location.id;
        acl_desc[i].location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    }
    result = npu_map_error(g_acl.p_aclrtMemSetAccess((void *)(uintptr_t)ptr, size,
                                                     acl_desc, count));
    free(acl_desc);
    return result;
}

CUresult CUDAAPI npu_cuMemUnmap(CUdeviceptr ptr, size_t size) {
    (void)size; /* aclrtUnmapMem removes the whole mapping at the exact base. */
    return npu_map_error(g_acl.p_aclrtUnmapMem((void *)(uintptr_t)ptr));
}

CUresult CUDAAPI npu_cuMemRelease(CUmemGenericAllocationHandle handle) {
    return npu_map_error(g_acl.p_aclrtFreePhysical((aclrtDrvMemHandle)(uintptr_t)handle));
}

CUresult CUDAAPI npu_cuMemcpyHtoDAsync(CUdeviceptr dst, const void *src, size_t bytes,
                                       CUstream hStream) {
    /* Success only means the copy was queued; aclrtSynchronizeStream is where
     * a failed transfer is reported, so the stream must be a real NPU stream. */
    return npu_map_error(g_acl.p_aclrtMemcpyAsync((void *)(uintptr_t)dst, bytes, src, bytes,
                                                  ACL_MEMCPY_HOST_TO_DEVICE, (aclrtStream)hStream));
}

CUresult CUDAAPI npu_cuEventCreate(CUevent *phEvent, unsigned int flags) {
    NpuEvent *event = calloc(1, sizeof(*event));
    aclError err;

    (void)flags; /* CUDA timing flags have no ACL equivalent; create a plain event. */
    if (!event) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    err = g_acl.p_aclrtCreateEvent(&event->event);
    if (err != ACL_SUCCESS) {
        free(event);
        return npu_map_error(err);
    }
    *phEvent = (CUevent)event;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI npu_cuEventDestroy(CUevent hEvent) {
    NpuEvent *event = (NpuEvent *)hEvent;
    CUresult result = npu_map_error(g_acl.p_aclrtDestroyEvent(event->event));

    free(event);
    return result;
}

CUresult CUDAAPI npu_cuEventRecord(CUevent hEvent, CUstream hStream) {
    NpuEvent *event = (NpuEvent *)hEvent;

    event->stream = (aclrtStream)hStream;
    return npu_map_error(g_acl.p_aclrtRecordEvent(event->event, event->stream));
}

CUresult CUDAAPI npu_cuEventSynchronize(CUevent hEvent) {
    NpuEvent *event = (NpuEvent *)hEvent;
    aclError err = g_acl.p_aclrtSynchronizeEvent(event->event);

    if (err == ACL_SUCCESS) {
        err = g_acl.p_aclrtSynchronizeStream(event->stream);
    }
    return npu_map_error(err);
}
