#pragma once

#if 0
#include "nccl.h"


namespace utils {
namespace detail {

  template <>
  ALWAYS_INLINE const char *error_string<ncclResult_t>(const ncclResult_t &status) {
    return ncclGetErrorString(status);
  }

  template <>
  ALWAYS_INLINE bool is_success<ncclResult_t>(const ncclResult_t &err) {
    return err == ncclSuccess;
  }

} // namespace detail
} // namespace utils

#endif