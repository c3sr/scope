
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <utility>
#include <vector>

#include "utils/compat.hpp"
#include "utils/error.hpp"

namespace utils {
namespace detail {

  static std::vector<std::pair<int, int>> nGpuArchCoresPerSM{{0x10, 8},   // Tesla Generation (SM 1.0) G80 class
                                                             {0x11, 8},   // Tesla Generation (SM 1.1) G8x class
                                                             {0x12, 8},   // Tesla Generation (SM 1.2) G9x class
                                                             {0x13, 8},   // Tesla Generation (SM 1.3) GT200 class
                                                             {0x20, 32},  // Fermi Generation (SM 2.0) GF100 class
                                                             {0x21, 48},  // Fermi Generation (SM 2.1) GF10x class
                                                             {0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
                                                             {0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
                                                             {0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
                                                             {0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
                                                             {0x60, 64},  // Pascal Generation (SM 6.0) GP100 class
                                                             {0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
                                                             {0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
                                                             {0x70, 64},  // Volta Generation (SM 7.0) GV100 class
                                                             {-1, -1}};

  template <>
  ALWAYS_INLINE const char *error_string<cudaError_t>(const cudaError_t &status) {
    return cudaGetErrorString(status);
  }

  template <>
  ALWAYS_INLINE bool is_success<cudaError_t>(const cudaError_t &err) {
    return err == cudaSuccess;
  }

  template <>
  ALWAYS_INLINE const char *error_string<CUresult>(const CUresult &status) {

    switch (status) {
      case CUDA_SUCCESS:
        return "Success";
      case CUDA_ERROR_ALREADY_ACQUIRED:
        return "CUDA_ERROR_ALREADY_ACQUIRED";
      case CUDA_ERROR_ALREADY_MAPPED:
        return "CUDA_ERROR_ALREADY_MAPPED";
      case CUDA_ERROR_ARRAY_IS_MAPPED:
        return "CUDA_ERROR_ARRAY_IS_MAPPED";
      case CUDA_ERROR_ASSERT:
        return "CUDA_ERROR_ASSERT";
      case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
        return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";
      case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
        return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";
      case CUDA_ERROR_CONTEXT_IS_DESTROYED:
        return "CUDA_ERROR_CONTEXT_IS_DESTROYED";
      case CUDA_ERROR_DEINITIALIZED:
        return "CUDA_ERROR_DEINITIALIZED";
      case CUDA_ERROR_ECC_UNCORRECTABLE:
        return "CUDA_ERROR_ECC_UNCORRECTABLE";
      case CUDA_ERROR_FILE_NOT_FOUND:
        return "CUDA_ERROR_FILE_NOT_FOUND";
      case CUDA_ERROR_HARDWARE_STACK_ERROR:
        return "CUDA_ERROR_HARDWARE_STACK_ERROR";
      case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
        return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";
      case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
        return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";
      case CUDA_ERROR_ILLEGAL_ADDRESS:
        return "CUDA_ERROR_ILLEGAL_ADDRESS";
      case CUDA_ERROR_ILLEGAL_INSTRUCTION:
        return "CUDA_ERROR_ILLEGAL_INSTRUCTION";
      case CUDA_ERROR_INVALID_ADDRESS_SPACE:
        return "CUDA_ERROR_INVALID_ADDRESS_SPACE";
      case CUDA_ERROR_INVALID_CONTEXT:
        return "CUDA_ERROR_INVALID_CONTEXT";
      case CUDA_ERROR_INVALID_DEVICE:
        return "CUDA_ERROR_INVALID_DEVICE";
      case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
        return "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT";
      case CUDA_ERROR_INVALID_HANDLE:
        return "CUDA_ERROR_INVALID_HANDLE";
      case CUDA_ERROR_INVALID_IMAGE:
        return "CUDA_ERROR_INVALID_IMAGE";
      case CUDA_ERROR_INVALID_PC:
        return "CUDA_ERROR_INVALID_PC";
      case CUDA_ERROR_INVALID_PTX:
        return "CUDA_ERROR_INVALID_PTX";
      case CUDA_ERROR_INVALID_SOURCE:
        return "CUDA_ERROR_INVALID_SOURCE";
      case CUDA_ERROR_INVALID_VALUE:
        return "CUDA_ERROR_INVALID_VALUE";
      case CUDA_ERROR_LAUNCH_FAILED:
        return "CUDA_ERROR_LAUNCH_FAILED";
      case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
        return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";
      case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
        return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
      case CUDA_ERROR_LAUNCH_TIMEOUT:
        return "CUDA_ERROR_LAUNCH_TIMEOUT";
      case CUDA_ERROR_MAP_FAILED:
        return "CUDA_ERROR_MAP_FAILED";
      case CUDA_ERROR_MISALIGNED_ADDRESS:
        return "CUDA_ERROR_MISALIGNED_ADDRESS";
      case CUDA_ERROR_NOT_FOUND:
        return "CUDA_ERROR_NOT_FOUND";
      case CUDA_ERROR_NOT_INITIALIZED:
        return "CUDA_ERROR_NOT_INITIALIZED";
      case CUDA_ERROR_NOT_MAPPED:
        return "CUDA_ERROR_NOT_MAPPED";
      case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
        return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";
      case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
        return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";
      case CUDA_ERROR_NOT_PERMITTED:
        return "CUDA_ERROR_NOT_PERMITTED";
      case CUDA_ERROR_NOT_READY:
        return "CUDA_ERROR_NOT_READY";
      case CUDA_ERROR_NOT_SUPPORTED:
        return "CUDA_ERROR_NOT_SUPPORTED";
      case CUDA_ERROR_NO_BINARY_FOR_GPU:
        return "CUDA_ERROR_NO_BINARY_FOR_GPU";
      case CUDA_ERROR_NO_DEVICE:
        return "CUDA_ERROR_NO_DEVICE";
      case CUDA_ERROR_NVLINK_UNCORRECTABLE:
        return "CUDA_ERROR_NVLINK_UNCORRECTABLE";
      case CUDA_ERROR_OPERATING_SYSTEM:
        return "CUDA_ERROR_OPERATING_SYSTEM";
      case CUDA_ERROR_OUT_OF_MEMORY:
        return "CUDA_ERROR_OUT_OF_MEMORY";
      case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
        return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";
      case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
        return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";
      case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
        return "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";
      case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
        return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";
      case CUDA_ERROR_PROFILER_ALREADY_STARTED:
        return "CUDA_ERROR_PROFILER_ALREADY_STARTED";
      case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
        return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";
      case CUDA_ERROR_PROFILER_DISABLED:
        return "CUDA_ERROR_PROFILER_DISABLED";
      case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
        return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";
      case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
        return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";
      case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
        return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";
      case CUDA_ERROR_TOO_MANY_PEERS:
        return "CUDA_ERROR_TOO_MANY_PEERS";
      case CUDA_ERROR_UNKNOWN:
        return "CUDA_ERROR_UNKNOWN";
      case CUDA_ERROR_UNMAP_FAILED:
        return "CUDA_ERROR_UNMAP_FAILED";
      case CUDA_ERROR_UNSUPPORTED_LIMIT:
        return "CUDA_ERROR_UNSUPPORTED_LIMIT";
      default:
        return "CUDA Driver Unknown Error";
    }
  }

  template <>
  ALWAYS_INLINE bool is_success<CUresult>(const CUresult &err) {
    return err == CUDA_SUCCESS;
  }




} // namespace detail

  ALWAYS_INLINE cudaError_t cuda_reset_device(const int &id) {
    cudaError_t err = cudaSetDevice(id);
    if (err != cudaSuccess) {
      return err;
    }
    return cudaDeviceReset();
  }

} // namespace utils
