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
    layer_benchmarks.hpp
)

sugar_files(
    BENCHMARK_SOURCES
    #   conv_bwd_bias.cpp
    #   batchnorm_bwd.cpp
    #   softmax_bwd.cpp
    #   ctc_loss.cpp
       softmax_fwd.cpp
       batchnorm_fwd.cpp
    #   element_wise.cpp
    #   dropout_bwd.cpp
    #   find_conv_alg.cpp
       activation_fwd.cpp
    #   conv_bwd_data.cpp
    #   activation_bwd.cpp
    #   pooling_bwd.cpp
       dropout_fwd.cpp
    #   conv_bias_activation_fwd.cpp
    #   scale.cpp
    #   reduce.cpp
       conv_fwd.cpp
    #   conv_bwd_filter.cpp
    #   find_rnn_alg.cpp
       pooling_fwd.cpp
)

#   sugar_files(
#       BENCHMARK_CUDA_SOURCES
#       cuda_conv_activation_lrn_pool_fused.cu
#       cuda_conv_activation_lrn_pool_basic.cu
#   )

