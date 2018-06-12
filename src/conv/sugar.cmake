# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED CONV_SUGAR_CMAKE_)
  return()
else()
  set(CONV_SUGAR_CMAKE_ 1)
endif()

include(sugar_files)

sugar_files(
    BENCHMARK_HEADERS
    args.hpp
    utils.hpp
)

sugar_files(
    BENCHMARK_SOURCES
    cudnn.cpp
)

sugar_files(
    BENCHMARK_CUDA_SOURCES
    cuda_conv_activation_lrn_pool_basic.cu
    cuda_conv_activation_lrn_pool_fused.cu
    cuda.cu
)

