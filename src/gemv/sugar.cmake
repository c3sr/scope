# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED BENCHMARK_SRC_GEMV_SUGAR_CMAKE_)
  return()
else()
  set(BENCHMARK_SRC_GEMV_SUGAR_CMAKE_ 1)
endif()

include(sugar_files)

sugar_files(
    BENCHMARK_SOURCES
    cblas.cpp
)

sugar_files(
    BENCHMARK_CUDA_SOURCES
    cuda.cu
)

