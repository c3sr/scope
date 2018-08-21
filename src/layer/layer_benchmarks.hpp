
#include <benchmark/benchmark.h>

#include <cmath>
#include <iostream>
#include <mutex>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <cudnn.h>

#include "init/init.hpp"
#include "utils/utils.hpp"

#include "layer/args.hpp"
#include "layer/helper.hpp"
#include "layer/utils.hpp"

template <typename T, cudnnConvolutionFwdAlgo_t convolution_algorithm
#ifdef CUDNN_SUPPORTS_TENSOR_OPS
          ,
          cudnnMathType_t math_type = CUDNN_DEFAULT_MATH
#endif
          >
extern void LAYER_CUDNN_CONV_FWD_Impl(benchmark::State& state);

template <typename T, cudnnBatchNormMode_t batchnorm_mode>
extern void LAYER_CUDNN_BATCHNORM_FWD_INFERENCE_Impl(benchmark::State& state);

template <typename T, cudnnBatchNormMode_t batchnorm_mode>
extern void LAYER_CUDNN_BATCHNORM_FWD_TRAINING_Impl(benchmark::State& state);
