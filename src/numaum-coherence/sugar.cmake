# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED SRC_NUMAUM_COHERENCE_SUGAR_CMAKE_)
  return()
else()
  set(SRC_NUMAUM_COHERENCE_SUGAR_CMAKE_ 1)
endif()

include(sugar_files)

sugar_files(
    BENCHMARK_HEADERS
    args.hpp
)

sugar_files(
    BENCHMARK_CUDA_SOURCES
    gpu_to_host.cu
    host_to_gpu.cu
    gpu_to_host_threads.cu
    gpu_threads.cu
)

