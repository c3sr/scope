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
    } else if (std::is_same<T, float>::value) {
      return "SGEMM";
    } else if (std::is_same<T, double>::value) {
      return "DGEMM";
    } else if (std::is_same<T, std::complex<float>>::value) {
      return "CGEMM";
    } else if (std::is_same<T, std::complex<double>>::value) {
      return "ZGEMM";
    } else {
      return "UNKNOWN_GEMM";
    }
  }

  template <typename T>
  static T one() {
    return T{1};
  };

  template <>
  __half one<__half>() {
    unsigned short x{1};
    __half res;
    memcpy(&res, &x, sizeof(res));
    return res;
  };

  template <typename T>
  static T zero() {
    return T{0};
  };

  template <>
  __half zero<__half>() {
    __half res;
    memset(&res, 0, sizeof(res));
    return res;
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
