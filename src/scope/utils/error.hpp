#pragma once

#include <iostream>
#include <string>

#include "scope/init/logger.hpp"
#include "scope/utils/compat.hpp"

namespace utils {
namespace detail {

  template <typename T>
  static ALWAYS_INLINE const char *error_string(const T &err);

  template <typename T>
  static ALWAYS_INLINE bool is_success(const T &err);

  template <typename T>
  static ALWAYS_INLINE bool is_error(const T &err) {
    return !is_success<T>(err);
  }

  template <typename T>
  static ALWAYS_INLINE bool print_if_error(const T &err, const char *stmt, const char *file, const char *func,
                                           int line) {
    if (is_success<T>(err)) {
      return false;
    }
#if defined(__CUDA_ARCH__)
    // in device code
    printf("ERROR on %s::%d In %s:(%s) FAILED", file, line, func, stmt);
#else  // defined(__CUDA_ARCH__)
    // in host code
    LOG(critical, "ERROR[{}] on {}::{} In {}:({}) FAILED", error_string<T>(err), file, line, func, stmt);
#endif // defined(__CUDA_ARCH__)
    return true;
  }

} // namespace detail
} // namespace utils

#ifndef IS_ERROR
#define IS_ERROR(stmt) utils::detail::is_error(stmt)
#endif // IS_ERROR

#ifndef PRINT_IF_ERROR
#define PRINT_IF_ERROR(stmt) utils::detail::print_if_error(stmt, #stmt, __FILE__, __func__, __LINE__)
#endif // PRINT_IF_ERROR
