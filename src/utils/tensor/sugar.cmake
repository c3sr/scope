# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED SRC_UTILS_TENSOR_SUGAR_CMAKE_)
  return()
else()
  set(SRC_UTILS_TENSOR_SUGAR_CMAKE_ 1)
endif()

include(sugar_files)

sugar_files(
    BENCHMARK_HEADERS
    tensor.hpp
    details.hpp
    shape.hpp
)

