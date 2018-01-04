#pragma once

#include <complex>
#include <type_traits>

#include <cuda_runtime.h>

namespace axpy {
namespace detail {

  template <typename T>
  static const char* implementation_name() {
    if (std::is_same_v<T, float>) {
      return "SAXPY";
    }
    if (std::is_same_v<T, double>) {
      return "DAXPY";
    }
    if (std::is_same_v<T, std::complex<float>>) {
      return "CAXPY";
    }
    if (std::is_same_v<T, std::complex<double>>) {
      return "ZAXPY";
    }
    return "UNKNOWN_AXPY";
  }

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
} // namespace axpy