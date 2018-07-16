#pragma once

#include <chrono>

#include "scope/utils/compat.hpp"

static ALWAYS_INLINE std::chrono::time_point<std::chrono::high_resolution_clock> now() {
  return std::chrono::high_resolution_clock::now();
}