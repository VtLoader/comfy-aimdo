/* Private ABI for the Ascend NPU backend.
 *
 * CANN's ACL runtime headers are not part of the build environment, so the
 * handful of ACL types, constants and entry points the backend needs are
 * duck-typed here, mirroring what src/gpu_abi.h does for the CUDA driver ABI.
 * src-npu/acl-shim.c adapts the CUDA-flavored dispatch contract from
 * src/gpu_dispatch.h onto raw aclrt* calls.
 */
#pragma once

#include "gpu_abi.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef int aclError;
typedef void *aclrtStream;
typedef void *aclrtEvent;
typedef void *aclrtDrvMemHandle;

/* Values from CANN include/acl/acl_rt.h, acl_base_rt.h and
 * include/acl/error_codes/rt_error_codes.h. */
#define ACL_SUCCESS                        0
#define ACL_ERROR_INVALID_PARAM            100000
#define ACL_ERROR_RT_PARAM_INVALID         107000
#define ACL_ERROR_RT_FEATURE_NOT_SUPPORT   207000
#define ACL_ERROR_RT_MEMORY_ALLOCATION     207001
#define ACL_ERROR_RT_TS_ERROR              507001
#define ACL_ERROR_RT_DRV_INTERNAL_ERROR    507899
#define ACL_MEM_MALLOC_HUGE_FIRST          0
#define ACL_MEMCPY_HOST_TO_DEVICE          1
#define ACL_MEMCPY_DEVICE_TO_DEVICE        3
/* aclrtGetMemInfo accepts ACL_HBM_MEM; allocation granularity queries and
 * aclrtMallocPhysical only accept HUGE(4)/NORMAL(5) on this platform. */
#define ACL_HBM_MEM                        1
#define ACL_HBM_MEM_HUGE                   4
#define ACL_MEM_LOCATION_TYPE_DEVICE       1
#define ACL_MEM_ALLOCATION_TYPE_PINNED     0
#define ACL_MEM_HANDLE_TYPE_NONE           0
#define ACL_HOST_REG_PINNED                0x10000000UL

/* Values from the Ascend driver's drvError.h. libascend_hal returns these
 * (0 on success) instead of ACL error codes. */
#define DRV_ERROR_NONE                     0
#define DRV_ERROR_INVALID_VALUE            3
#define DRV_ERROR_OUT_OF_MEMORY            6
#define DRV_ERROR_PARA_ERROR               8
#define DRV_ERROR_NO_RESOURCES             14

typedef struct {
    uint32_t id;
    int type;
} AclMemLocation;

typedef struct {
    int flags;
    AclMemLocation location;
    uint8_t rsv[12];
} AclMemAccessDesc;

typedef struct {
    int handleType;
    int allocationType;
    int memAttr;
    AclMemLocation location;
    uint64_t reserve;
} AclPhysicalMemProp;

typedef aclError (*PFN_aclrtReserveMemAddress)(void **virPtr, size_t size,
                                               size_t alignment, void *expectPtr,
                                               uint64_t flags);
typedef aclError (*PFN_aclrtReleaseMemAddress)(void *virPtr);
typedef aclError (*PFN_aclrtMallocPhysical)(aclrtDrvMemHandle *handle, size_t size,
                                            const AclPhysicalMemProp *prop, uint64_t flags);
typedef aclError (*PFN_aclrtFreePhysical)(aclrtDrvMemHandle handle);
typedef aclError (*PFN_aclrtMapMem)(void *virPtr, size_t size, size_t offset,
                                    aclrtDrvMemHandle handle, uint64_t flags);
typedef aclError (*PFN_aclrtUnmapMem)(void *virPtr);
typedef aclError (*PFN_aclrtMemSetAccess)(void *virPtr, size_t size,
                                          AclMemAccessDesc *desc, size_t count);
typedef aclError (*PFN_aclrtMallocHost)(void **hostPtr, size_t size);
typedef aclError (*PFN_aclrtFreeHost)(void *hostPtr);
typedef aclError (*PFN_aclrtHostRegister)(void *ptr, uint64_t size, uint32_t flag);
typedef aclError (*PFN_aclrtHostUnregister)(void *ptr);
typedef aclError (*PFN_aclrtMemcpyAsync)(void *dst, size_t destMax, const void *src,
                                         size_t count, int kind, aclrtStream stream);
typedef aclError (*PFN_aclrtSynchronizeStream)(aclrtStream stream);
typedef aclError (*PFN_aclrtGetMemInfo)(int attr, size_t *free, size_t *total);
typedef aclError (*PFN_aclrtGetDevice)(int32_t *deviceId);
typedef aclError (*PFN_aclrtGetDeviceCount)(uint32_t *count);
typedef aclError (*PFN_aclrtSetDevice)(int32_t deviceId);
typedef aclError (*PFN_aclrtSynchronizeDevice)(void);
typedef aclError (*PFN_aclrtDeviceCanAccessPeer)(int32_t *canAccessPeer, int32_t deviceId,
                                                 int32_t peerDeviceId);
typedef aclError (*PFN_aclrtDeviceEnablePeerAccess)(int32_t peerDeviceId, uint32_t flags);
typedef aclError (*PFN_aclrtDeviceDisablePeerAccess)(int32_t peerDeviceId);
typedef aclError (*PFN_aclrtCreateEvent)(aclrtEvent *event);
typedef aclError (*PFN_aclrtRecordEvent)(aclrtEvent event, aclrtStream stream);
typedef aclError (*PFN_aclrtSynchronizeEvent)(aclrtEvent event);
typedef aclError (*PFN_aclrtDestroyEvent)(aclrtEvent event);
typedef const char *(*PFN_aclrtGetSocName)(void);
typedef aclError (*PFN_aclrtDeviceGetUuid)(int32_t deviceId, void *uuid);
typedef aclError (*PFN_aclrtMalloc)(void **devPtr, size_t size, int policy);
typedef aclError (*PFN_aclrtMallocAlign32)(void **devPtr, size_t size, int policy);
typedef aclError (*PFN_aclrtFree)(void *devPtr);
typedef const char *(*PFN_aclGetRecentErrMsg)(void);
typedef int (*PFN_halMemAddressReserve)(void **ptr, size_t size, size_t alignment,
                                        void *addr, uint64_t flag);
typedef int (*PFN_halMemAddressFree)(void *ptr);

typedef struct AimdoAclDispatch {
    PFN_aclrtReserveMemAddress p_aclrtReserveMemAddress;
    PFN_aclrtReleaseMemAddress p_aclrtReleaseMemAddress;
    PFN_aclrtMallocPhysical p_aclrtMallocPhysical;
    PFN_aclrtFreePhysical p_aclrtFreePhysical;
    PFN_aclrtMapMem p_aclrtMapMem;
    PFN_aclrtUnmapMem p_aclrtUnmapMem;
    PFN_aclrtMemSetAccess p_aclrtMemSetAccess;
    PFN_aclrtMallocHost p_aclrtMallocHost;
    PFN_aclrtFreeHost p_aclrtFreeHost;
    PFN_aclrtHostRegister p_aclrtHostRegister;
    PFN_aclrtHostUnregister p_aclrtHostUnregister;
    PFN_aclrtMemcpyAsync p_aclrtMemcpyAsync;
    PFN_aclrtSynchronizeStream p_aclrtSynchronizeStream;
    PFN_aclrtGetMemInfo p_aclrtGetMemInfo;
    PFN_aclrtGetDevice p_aclrtGetDevice;
    PFN_aclrtGetDeviceCount p_aclrtGetDeviceCount;
    PFN_aclrtSynchronizeDevice p_aclrtSynchronizeDevice;
    PFN_aclrtCreateEvent p_aclrtCreateEvent;
    PFN_aclrtRecordEvent p_aclrtRecordEvent;
    PFN_aclrtSynchronizeEvent p_aclrtSynchronizeEvent;
    PFN_aclrtDestroyEvent p_aclrtDestroyEvent;
    PFN_aclrtGetSocName p_aclrtGetSocName;
    PFN_aclrtMalloc p_aclrtMalloc;
    PFN_aclrtMallocAlign32 p_aclrtMallocAlign32;
    PFN_aclrtFree p_aclrtFree;
    PFN_aclrtDeviceGetUuid p_aclrtDeviceGetUuid;
    PFN_aclGetRecentErrMsg p_aclGetRecentErrMsg;
    PFN_halMemAddressReserve p_halMemAddressReserve;
    PFN_halMemAddressFree p_halMemAddressFree;
    bool hal_mem_address_available;
    /* Appended after the existing layout: external g_acl readers index the
     * halMemAddress* slots by position. */
    PFN_aclrtSetDevice p_aclrtSetDevice;
    PFN_aclrtDeviceCanAccessPeer p_aclrtDeviceCanAccessPeer;
    PFN_aclrtDeviceEnablePeerAccess p_aclrtDeviceEnablePeerAccess;
    PFN_aclrtDeviceDisablePeerAccess p_aclrtDeviceDisablePeerAccess;
} AimdoAclDispatch;

extern AimdoAclDispatch g_acl;

/* Host registration range table: npu_cuMemHostUnregister must not reach
 * aclrtHostUnregister for a range that was never registered (it faults when
 * the runtime is not initialized and errors on a stale pointer). The shim
 * updates the table for its own calls; src-posix/npu-funchooks.c updates it
 * for registrations issued directly against libascendcl. */
void npu_hostreg_init(void);
void npu_hostreg_cleanup(void);
void npu_hostreg_add(void *ptr, size_t size);
void npu_hostreg_remove(void *ptr);
bool npu_hostreg_contains(void *ptr);

/* Entry points wired into the shared AimdoCudaDispatch table. */
CUresult CUDAAPI npu_cuInit(unsigned int flags);
CUresult CUDAAPI npu_cuGetErrorString(CUresult error, const char **pStr);
CUresult CUDAAPI npu_cuCtxGetDevice(CUdevice *device);
CUresult CUDAAPI npu_cuCtxSynchronize(void);
CUresult CUDAAPI npu_cuDeviceGet(CUdevice *device, int ordinal);
CUresult CUDAAPI npu_cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
CUresult CUDAAPI npu_cuDeviceTotalMem(size_t *bytes, CUdevice dev);
CUresult CUDAAPI npu_cuDeviceGetName(char *name, int len, CUdevice dev);
CUresult CUDAAPI npu_cuDeviceGetUuid(CUuuid *uuid, CUdevice dev);
CUresult CUDAAPI npu_cuMemGetInfo(size_t *free_bytes, size_t *total_bytes);
CUresult CUDAAPI npu_cuMemAlloc(CUdeviceptr *dptr, size_t size);
CUresult CUDAAPI npu_cuMemFree(CUdeviceptr dptr);
CUresult CUDAAPI npu_cuMemAllocHost(void **pp, size_t bytesize);
CUresult CUDAAPI npu_cuMemFreeHost(void *p);
CUresult CUDAAPI npu_cuMemHostRegister(void *p, size_t bytesize, unsigned int flags);
CUresult CUDAAPI npu_cuMemHostUnregister(void *p);
CUresult CUDAAPI npu_cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment,
                                         CUdeviceptr addr, unsigned long long flags);
CUresult CUDAAPI npu_cuMemAddressFree(CUdeviceptr ptr, size_t size);
CUresult CUDAAPI npu_cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                                 const CUmemAllocationProp *prop, unsigned long long flags);
CUresult CUDAAPI npu_cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                              CUmemGenericAllocationHandle handle, unsigned long long flags);
CUresult CUDAAPI npu_cuMemSetAccess(CUdeviceptr ptr, size_t size,
                                    const CUmemAccessDesc *desc, size_t count);
CUresult CUDAAPI npu_cuMemUnmap(CUdeviceptr ptr, size_t size);
CUresult CUDAAPI npu_cuMemRelease(CUmemGenericAllocationHandle handle);
CUresult CUDAAPI npu_cuMemcpyHtoDAsync(CUdeviceptr dst, const void *src, size_t bytes,
                                       CUstream hStream);
CUresult CUDAAPI npu_cuEventCreate(CUevent *phEvent, unsigned int flags);
CUresult CUDAAPI npu_cuEventDestroy(CUevent hEvent);
CUresult CUDAAPI npu_cuEventRecord(CUevent hEvent, CUstream hStream);
CUresult CUDAAPI npu_cuEventSynchronize(CUevent hEvent);
