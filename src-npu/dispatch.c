/* Ascend NPU backend's implementation of the GPU dispatch contract
 * (src/gpu_dispatch.h).
 *
 * Unlike CUDA/HIP there is no vendor get-proc-address entry point, so the
 * aclrt* symbols are resolved by name from libascendcl and stored in g_acl.
 * src-npu/acl-shim.c then adapts them to the CUDA-shaped g_cuda table.
 */
#define _GNU_SOURCE

#include "plat.h"
#include "acl-shim.h"

#include <dlfcn.h>

static void *g_acl_module;
static void *g_hal_module;

/* HostBuffer reaches cuMemHostRegister/Unregister before and after plat_init
 * (users may build a HostBuffer after control.init without init_devices). Keep
 * the shim adapters wired from load time so a zeroed g_acl degrades to a
 * guarded no-op instead of a call through a NULL g_cuda slot. */
AimdoCudaDispatch g_cuda = {
    .p_cuMemHostRegister = npu_cuMemHostRegister,
    .p_cuMemHostUnregister = npu_cuMemHostUnregister,
};

typedef struct {
    void **slot;
    const char *symbol;
} DispatchSymbol;

static const DispatchSymbol dispatch_symbols[] = {
    { (void **)&g_acl.p_aclrtReserveMemAddress, "aclrtReserveMemAddress" },
    { (void **)&g_acl.p_aclrtReleaseMemAddress, "aclrtReleaseMemAddress" },
    { (void **)&g_acl.p_aclrtMallocPhysical, "aclrtMallocPhysical" },
    { (void **)&g_acl.p_aclrtFreePhysical, "aclrtFreePhysical" },
    { (void **)&g_acl.p_aclrtMapMem, "aclrtMapMem" },
    { (void **)&g_acl.p_aclrtUnmapMem, "aclrtUnmapMem" },
    { (void **)&g_acl.p_aclrtMemSetAccess, "aclrtMemSetAccess" },
    { (void **)&g_acl.p_aclrtMallocHost, "aclrtMallocHost" },
    { (void **)&g_acl.p_aclrtFreeHost, "aclrtFreeHost" },
    { (void **)&g_acl.p_aclrtHostRegister, "aclrtHostRegisterV2" },
    { (void **)&g_acl.p_aclrtHostUnregister, "aclrtHostUnregister" },
    { (void **)&g_acl.p_aclrtMemcpyAsync, "aclrtMemcpyAsync" },
    { (void **)&g_acl.p_aclrtSynchronizeStream, "aclrtSynchronizeStream" },
    { (void **)&g_acl.p_aclrtGetMemInfo, "aclrtGetMemInfo" },
    { (void **)&g_acl.p_aclrtGetDevice, "aclrtGetDevice" },
    { (void **)&g_acl.p_aclrtGetDeviceCount, "aclrtGetDeviceCount" },
    { (void **)&g_acl.p_aclrtSetDevice, "aclrtSetDevice" },
    { (void **)&g_acl.p_aclrtSynchronizeDevice, "aclrtSynchronizeDevice" },
    { (void **)&g_acl.p_aclrtCreateEvent, "aclrtCreateEvent" },
    { (void **)&g_acl.p_aclrtRecordEvent, "aclrtRecordEvent" },
    { (void **)&g_acl.p_aclrtSynchronizeEvent, "aclrtSynchronizeEvent" },
    { (void **)&g_acl.p_aclrtDestroyEvent, "aclrtDestroyEvent" },
    { (void **)&g_acl.p_aclrtGetSocName, "aclrtGetSocName" },
    { (void **)&g_acl.p_aclrtMalloc, "aclrtMalloc" },
    { (void **)&g_acl.p_aclrtFree, "aclrtFree" },
};

static const DispatchSymbol optional_symbols[] = {
    { (void **)&g_acl.p_aclrtDeviceGetUuid, "aclrtDeviceGetUuid" },
    { (void **)&g_acl.p_aclrtMallocAlign32, "aclrtMallocAlign32" },
    { (void **)&g_acl.p_aclGetRecentErrMsg, "aclGetRecentErrMsg" },
};

static const char *const acl_library_names[] = {
    "libascendcl.so",
    "libascendcl.so.1",
};

/* halMemAddressReserve/Free in libascend_hal reserve VA in arbitrary sizes,
 * unlike aclrtReserveMemAddress which rejects a single request >= 2 GiB. The
 * real driver library is preferred; the CANN devlib stub is only a fallback. */
static const char *const hal_library_paths[] = {
    "/usr/local/Ascend/driver/lib64/driver/libascend_hal.so",
    "/usr/local/Ascend/cann-9.0.0/aarch64-linux/devlib/libascend_hal.so",
};

static void aimdo_hal_resolve_address_api(void) {
    g_acl.p_halMemAddressReserve = (PFN_halMemAddressReserve)dlsym(RTLD_DEFAULT, "halMemAddressReserve");
    g_acl.p_halMemAddressFree = (PFN_halMemAddressFree)dlsym(RTLD_DEFAULT, "halMemAddressFree");

    if (g_acl.p_halMemAddressReserve && g_acl.p_halMemAddressFree) {
        g_acl.hal_mem_address_available = true;
        return;
    }

    for (size_t i = 0; i < ARRAY_SIZE(hal_library_paths); i++) {
        void *module = dlopen(hal_library_paths[i], RTLD_LAZY | RTLD_LOCAL);
        void *reserve;
        void *release;

        if (!module) {
            continue;
        }
        reserve = dlsym(module, "halMemAddressReserve");
        release = dlsym(module, "halMemAddressFree");
        if (reserve && release) {
            g_acl.p_halMemAddressReserve = (PFN_halMemAddressReserve)reserve;
            g_acl.p_halMemAddressFree = (PFN_halMemAddressFree)release;
            g_acl.hal_mem_address_available = true;
            g_hal_module = module;
            log(INFO, "%s: loaded %s\n", __func__, hal_library_paths[i]);
            return;
        }
        dlclose(module);
    }

    g_acl.p_halMemAddressReserve = NULL;
    g_acl.p_halMemAddressFree = NULL;
    g_acl.hal_mem_address_available = false;
    log(WARNING, "%s: halMemAddress* not resolved, VA reservations limited to <2GiB\n", __func__);
}

static void *aimdo_acl_dlopen(void) {
    void *loaded = aimdo_find_loaded_module(acl_library_names, ARRAY_SIZE(acl_library_names));

    if (loaded) {
        return loaded;
    }
    for (size_t i = 0; i < ARRAY_SIZE(acl_library_names); i++) {
        void *module = dlopen(acl_library_names[i], RTLD_LAZY | RTLD_LOCAL);

        if (module) {
            log(INFO, "%s: loaded %s\n", __func__, acl_library_names[i]);
            return module;
        }
    }
    return NULL;
}

static void aimdo_acl_wire_dispatch(void) {
    g_cuda.p_cuInit = npu_cuInit;
    g_cuda.p_cuGetErrorString = npu_cuGetErrorString;
    g_cuda.p_cuCtxGetDevice = npu_cuCtxGetDevice;
    g_cuda.p_cuCtxSynchronize = npu_cuCtxSynchronize;
    g_cuda.p_cuDeviceGet = npu_cuDeviceGet;
    g_cuda.p_cuDeviceGetAttribute = npu_cuDeviceGetAttribute;
    g_cuda.p_cuDeviceTotalMem = npu_cuDeviceTotalMem;
    g_cuda.p_cuDeviceGetName = npu_cuDeviceGetName;
    g_cuda.p_cuDeviceGetUuid = npu_cuDeviceGetUuid;
    g_cuda.p_cuMemGetInfo = npu_cuMemGetInfo;
    g_cuda.p_cuMemAlloc_v2 = npu_cuMemAlloc;
    g_cuda.p_cuMemFree_v2 = npu_cuMemFree;
    g_cuda.p_cuMemAllocHost = npu_cuMemAllocHost;
    g_cuda.p_cuMemFreeHost = npu_cuMemFreeHost;
    g_cuda.p_cuMemHostRegister = npu_cuMemHostRegister;
    g_cuda.p_cuMemHostUnregister = npu_cuMemHostUnregister;
    g_cuda.p_cuMemAddressReserve = npu_cuMemAddressReserve;
    g_cuda.p_cuMemAddressFree = npu_cuMemAddressFree;
    g_cuda.p_cuMemCreate = npu_cuMemCreate;
    g_cuda.p_cuMemMap = npu_cuMemMap;
    g_cuda.p_cuMemSetAccess = npu_cuMemSetAccess;
    g_cuda.p_cuMemUnmap = npu_cuMemUnmap;
    g_cuda.p_cuMemRelease = npu_cuMemRelease;
    g_cuda.p_cuMemcpyHtoDAsync = npu_cuMemcpyHtoDAsync;
    g_cuda.p_cuEventCreate = npu_cuEventCreate;
    g_cuda.p_cuEventDestroy = npu_cuEventDestroy;
    g_cuda.p_cuEventRecord = npu_cuEventRecord;
    g_cuda.p_cuEventSynchronize = npu_cuEventSynchronize;
}

bool aimdo_cuda_runtime_init(void) {
    if (g_cuda.p_cuInit) {
        return true;
    }

    g_acl_module = aimdo_acl_dlopen();
    if (!g_acl_module) {
        log(AIMDO_LOG_ERROR, "%s: failed to load libascendcl\n", __func__);
        return false;
    }

    for (size_t i = 0; i < ARRAY_SIZE(dispatch_symbols); i++) {
        void *resolved = dlsym(g_acl_module, dispatch_symbols[i].symbol);

        if (!resolved) {
            log(AIMDO_LOG_ERROR, "%s: failed to resolve required ACL symbol %s\n", __func__,
                dispatch_symbols[i].symbol);
            aimdo_cuda_runtime_cleanup();
            return false;
        }
        *dispatch_symbols[i].slot = resolved;
    }

    for (size_t i = 0; i < ARRAY_SIZE(optional_symbols); i++) {
        *optional_symbols[i].slot = dlsym(g_acl_module, optional_symbols[i].symbol);
    }

    aimdo_hal_resolve_address_api();
    aimdo_acl_wire_dispatch();
    npu_hostreg_init();

    if (g_cuda.p_cuInit(0) != CUDA_SUCCESS) {
        log(AIMDO_LOG_ERROR, "%s: ACL runtime init failed\n", __func__);
        aimdo_cuda_runtime_cleanup();
        return false;
    }

    return true;
}

void aimdo_cuda_runtime_cleanup(void) {
    void *acl_module = g_acl_module;
    void *hal_module = g_hal_module;

    g_acl_module = NULL;
    g_hal_module = NULL;
    npu_hostreg_cleanup();
    memset(&g_cuda, 0, sizeof(g_cuda));
    memset(&g_acl, 0, sizeof(g_acl));
    g_cuda.p_cuMemHostRegister = npu_cuMemHostRegister;
    g_cuda.p_cuMemHostUnregister = npu_cuMemHostUnregister;

    if (hal_module) {
        dlclose(hal_module);
    }
    if (acl_module) {
        dlclose(acl_module);
    }
}
