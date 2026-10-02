/* Linux (funchook) installer for the Ascend NPU build, mirroring
 * src-posix/cuda-funchooks.c but hooking libascendcl instead of libcuda.
 * The aclrt* targets were resolved by src-npu/dispatch.c into g_acl; each
 * installed trampoline replaces the slot in place so the shim adapters keep
 * calling the real allocator.
 */
#define _GNU_SOURCE

#include "plat.h"
#include "acl-shim.h"

#include <funchook.h>

static funchook_t *funchook_state;

static int npu_alloc_accounted(void **devPtr, size_t size) {
    int status = aimdo_cuda_malloc((CUdeviceptr *)devPtr, size, npu_cuMemAlloc);

    return status == CUDA_ERROR_OUT_OF_MEMORY ? ACL_ERROR_RT_MEMORY_ALLOCATION : status;
}

static int npu_hook_aclrtMalloc(void **devPtr, size_t size, int policy) {
    (void)policy; /* torch_npu allocates with ACL_MEM_MALLOC_HUGE_FIRST. */
    return npu_alloc_accounted(devPtr, size);
}

static int npu_hook_aclrtMallocAlign32(void **devPtr, size_t size, int policy) {
    (void)policy;
    return npu_alloc_accounted(devPtr, size);
}

static int npu_hook_aclrtFree(void *devPtr) {
    return aimdo_cuda_free((CUdeviceptr)(uintptr_t)devPtr, npu_cuMemFree);
}

/* ComfyUI pins host memory by calling the ACL host register entry point
 * directly, so the raw ACL entry points update the shim's registration table. */
static int npu_hook_aclrtHostRegister(void *ptr, uint64_t size, uint32_t flag) {
    int status = g_acl.p_aclrtHostRegister(ptr, size, flag);

    if (status == ACL_SUCCESS) {
        npu_hostreg_add(ptr, (size_t)size);
    }
    return status;
}

static int npu_hook_aclrtHostUnregister(void *ptr) {
    int status = g_acl.p_aclrtHostUnregister(ptr);

    if (status == ACL_SUCCESS) {
        npu_hostreg_remove(ptr);
    }
    return status;
}

typedef struct {
    void **target;
    void *hook;
    const char *name;
} NpuHookEntry;

static const NpuHookEntry hooks[] = {
    { (void **)&g_acl.p_aclrtMalloc, (void *)npu_hook_aclrtMalloc, "aclrtMalloc" },
    { (void **)&g_acl.p_aclrtMallocAlign32, (void *)npu_hook_aclrtMallocAlign32, "aclrtMallocAlign32" },
    { (void **)&g_acl.p_aclrtFree, (void *)npu_hook_aclrtFree, "aclrtFree" },
    { (void **)&g_acl.p_aclrtHostRegister, (void *)npu_hook_aclrtHostRegister, "aclrtHostRegisterV2" },
    { (void **)&g_acl.p_aclrtHostUnregister, (void *)npu_hook_aclrtHostUnregister, "aclrtHostUnregister" },
};

bool aimdo_setup_hooks(void) {
    int installed = 0;

    if (!g_acl.p_aclrtMalloc || !g_acl.p_aclrtFree) {
        log(AIMDO_LOG_ERROR, "%s: ACL allocator entry points are not resolved\n", __func__);
        return false;
    }

    funchook_state = funchook_create();
    if (!funchook_state) {
        log(AIMDO_LOG_ERROR, "%s: funchook_create failed\n", __func__);
        return false;
    }

    for (size_t i = 0; i < ARRAY_SIZE(hooks); i++) {
        const char *detail;
        int status;

        if (!*hooks[i].target) {
            log(WARNING, "%s: %s not available, skipping hook\n", __func__, hooks[i].name);
            continue;
        }
        status = funchook_prepare(funchook_state, hooks[i].target, hooks[i].hook);
        if (status != FUNCHOOK_ERROR_SUCCESS) {
            detail = funchook_error_message(funchook_state);
            log(WARNING, "%s: funchook_prepare(%s) failed: %d %s\n", __func__, hooks[i].name,
                status, detail ? detail : "<unknown funchook error>");
            continue;
        }
        installed++;
    }

    {
        int status = funchook_install(funchook_state, 0);

        if (status != FUNCHOOK_ERROR_SUCCESS) {
            const char *detail = funchook_error_message(funchook_state);

            log(AIMDO_LOG_ERROR, "%s: funchook_install failed: %d %s\n", __func__, status,
                detail ? detail : "<unknown funchook error>");
            goto fail_teardown;
        }
    }

    log(DEBUG, "%s: installed %d ACL allocator hooks\n", __func__, installed);
    return true;

fail_teardown:
    aimdo_teardown_hooks();
    return false;
}

void aimdo_teardown_hooks(void) {
    int status;

    if (!funchook_state) {
        return;
    }

    if (((status = funchook_uninstall(funchook_state, 0)) != FUNCHOOK_ERROR_SUCCESS &&
         status != FUNCHOOK_ERROR_NOT_INSTALLED) ||
        (status = funchook_destroy(funchook_state)) != FUNCHOOK_ERROR_SUCCESS) {
        const char *detail = funchook_error_message(funchook_state);

        log(AIMDO_LOG_ERROR, "%s: funchook teardown failed: %d %s\n", __func__, status,
            detail ? detail : "<unknown funchook error>");
    }

    funchook_state = NULL;
}
