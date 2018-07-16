# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED SCOPE_UTILS_TENSOR_SUGAR_CMAKE_)
  return()
else()
  set(SCOPE_UTILS_TENSOR_SUGAR_CMAKE_ 1)
endif()

include(sugar_files)

sugar_files(
    BENCHMARK_HEADERS
    details.hpp
    shape.hpp
    tensor.hpp
)

