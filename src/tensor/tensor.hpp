#pragma once

#include "tensor/detail.hpp"
#include "utils/utils.hpp"
#include <array>

namespace tensor {
  template <index_type... Dims>
  struct shape : public non_copyable {

    static constexpr index_type rank         = sizeof...(Dims);
    static constexpr auto dims               = std::array<index_type, rank>{Dims...};
    static constexpr size_t flattened_length = mpl::mul<Dims...>::value;
  };
}