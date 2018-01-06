#pragma once

#include <complex>
#include <type_traits>

#include <cuda_runtime.h>

namespace conv {
namespace detail {

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
} // namespace detail
} // namespace conv