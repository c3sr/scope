
#pragma once

#include <cuda_runtime.h>

#include "utils/compat.hpp"
#include "utils/error.hpp"

namespace utils {
namespace detail {

  template <>
  ALWAYS_INLINE const char *error_string<cudaError_t>(const cudaError_t &status) {
    return cudaGetErrorString(status);
  }

  template <>
  ALWAYS_INLINE bool is_success<cudaError_t>(const cudaError_t &err) {
    return err == cudaSuccess;
  }

} // namespace detail
} // namespace utils
