#pragma once

#include <complex>
#include <type_traits>

#include <cuda_runtime.h>

namespace gemm {
namespace detail {

  template <typename T>
  static const char* implementation_name() {
    if (std::is_same<T, __half>::value) {
      return "HGEMM";
    }
    if (std::is_same<T, float>::value) {
      return "SGEMM";
    }
    if (std::is_same<T, double>::value) {
      return "DGEMM";
    }
    if (std::is_same<T, std::complex<float>>::value) {
      return "CGEMM";
    }
    if (std::is_same<T, std::complex<double>>::value) {
      return "ZGEMM";
    }
    return "UNKNOWN_GEMM";
  }

  template <typename T>
  static T one() {
    return T{1};
  };

  template <>
  __half one<__half>() {
    const __half_raw x{1};
    return __half{x};
  };

  template <typename T>
  static T zero() {
    return T{0};
  };

  template <>
  __half zero<__half>() {
    const __half_raw x{1};
    return __half{x};
  };

  template <typename T>
  struct cuda_type {
    using type = T;
  };
  template <>
  struct cuda_type<std::complex<float>> {
    using type = cuComplex;
  };
  template <>
  struct cuda_type<std::complex<double>> {
    using type = cuDoubleComplex;
  };

} // namespace detail
} // namespace gemm