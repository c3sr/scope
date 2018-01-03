#pragma once

#include <complex>
#include <type_traits>

#include <cuda_runtime.h>

namespace gemm {
namespace detail {

  template <typename T>
  const char* implementation_name() {
    if constexpr (std::is_same_v<T, float>) {
      return "SGEMM";
    }
    if constexpr (std::is_same_v<T, double>) {
      return "DGEMM";
    }
    if constexpr (std::is_same_v<T, std::complex<float>>) {
      return "CGEMM";
    }
    if constexpr (std::is_same_v<T, std::complex<double>>) {
      return "ZGEMM";
    }
    return "UNKNOWN_GEMM";
  }

  template <class T>
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