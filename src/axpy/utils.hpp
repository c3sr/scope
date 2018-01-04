#pragma once

#include <complex>
#include <type_traits>

#include <cuda_runtime.h>

namespace axpy {
namespace detail {

  template <typename T>
  static const char* implementation_name() {
    if (std::is_same<T, float>::value) {
      return "SAXPY";
    }
    if (std::is_same<T, double>::value) {
      return "DAXPY";
    }
    if (std::is_same<T, std::complex<float>>::value) {
      return "CAXPY";
    }
    if (std::is_same<T, std::complex<double>>::value) {
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