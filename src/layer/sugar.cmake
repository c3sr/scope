# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED SRC_LAYER_SUGAR_CMAKE_)
  return()
else()
  set(SRC_LAYER_SUGAR_CMAKE_ 1)
endif()

include(sugar_files)

sugar_files(
    BENCHMARK_HEADERS
    helper.hpp
    utils.hpp
    args.hpp
)

sugar_files(
    BENCHMARK_SOURCES
    batchnorm_bwd.cpp
    softmax_bwd.cpp
    ctc_loss.cpp
    softmax_fwd.cpp
    batchnorm_fwd.cpp
    element_wise.cpp
    pooling.cpp
    find_conv_alg.cpp
    activation_fwd.cpp
    activation_bwd.cpp
    scale.cpp
    reduce.cpp
    conv_fwd.cpp
    conv_fwd_get_algo
    conv_bwd.cpp
    find_rnn_alg.cpp
)

sugar_files(
    BENCHMARK_CUDA_SOURCES
    cuda_conv_activation_lrn_pool_fused.cu
    cuda_conv_activation_lrn_pool_basic.cu
)

