#define BENCHMARK_NAME "CUDNN/CONV_FWD_GET_ALGO"

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

// http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
// http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionFwdAlgo_t
template <typename T
#ifdef CUDNN_SUPPORTS_TENSOR_OPS
          ,
          cudnnMathType_t math_type = CUDNN_DEFAULT_MATH
#endif
          >
static void CUDNN_Impl(benchmark::State& state) {
  if (!has_cuda) {
    state.SkipWithError(BENCHMARK_NAME " no CUDA device found");
    return;
  }
#ifdef CUDNN_SUPPORTS_TENSOR_OPS
  if (math_type == CUDNN_TENSOR_OP_MATH && !detail::SupportsTensorCore(FLAG(cuda_device_id))) {
    state.SkipWithError(BENCHMARK_NAME "no Tensorcore support on current device");
    return;
  }
#endif // CUDNN_SUPPORTS_TENSOR_OPS

  const float alpha = 1, beta = 0;
  const cudnnConvolutionMode_t conv_mode = CUDNN_CONVOLUTION;

  //  w, h, c, n, k, filter_w(s), filter_h(r), pad_w, pad_h, wstride, hstride
  const auto width         = state.range(0);
  const auto height        = state.range(1);
  const auto channels      = state.range(2);
  const auto batch_size    = state.range(3);
  const auto kernel_size   = state.range(4);
  const auto filter_width  = state.range(5);
  const auto filter_height = state.range(6);
  const auto pad_width     = state.range(7);
  const auto pad_height    = state.range(8);
  const auto stride_width  = state.range(9);
  const auto stride_height = state.range(10);

  const int input_image_bytes = batch_size * channels * height * width * sizeof(T);
  const int kernel_bytes      = kernel_size * channels * filter_height * filter_width * sizeof(T);

  const auto N = batch_size, K = kernel_size, C = channels, H = height, W = width, R = filter_height, S = filter_width;

  // const auto output_width  = calc_out_dim(width, filter_width, pad_width, stride_width);
  // const auto output_height = calc_out_dim(height, filter_height.fh, pad_height, stride_height);

  auto input_image = std::vector<T>(input_image_bytes / sizeof(T));
  auto kernel      = std::vector<T>(kernel_bytes / sizeof(T));

  std::fill(input_image.begin(), input_image.end(), detail::one<T>());
  std::fill(kernel.begin(), kernel.end(), detail::one<T>());

  auto input_tensor = Tensor<T>(state,
                                {/*batch_size=*/batch_size,
                                 /*channels=*/channels,
                                 /*image_height=*/height,
                                 /*image_width=*/width});
  if (!input_tensor.is_valid) {
    return;
  }
  cudnnTensorDescriptor_t input_descriptor = input_tensor.get();

  const auto kernel_filter = Filter<T>(state,
                                       {/*out_channels=*/kernel_size,
                                        /*in_channels=*/channels,
                                        /*kernel_height=*/filter_height,
                                        /*kernel_width=*/filter_width});
  if (!kernel_filter.is_valid) {
    return;
  }
  cudnnFilterDescriptor_t kernel_descriptor = kernel_filter.get();

  cudnnConvolutionDescriptor_t convolution_descriptor;
  if (PRINT_IF_ERROR(cudnnCreateConvolutionDescriptor(&convolution_descriptor))) {
    state.SkipWithError(BENCHMARK_NAME " failed to cudnnCreateConvolutionDescriptor");
    return;
  }
  if (PRINT_IF_ERROR(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                                     /*pad_height=*/pad_height,
                                                     /*pad_width=*/pad_width,
                                                     /*vertical_stride=*/stride_height,
                                                     /*horizontal_stride=*/stride_width,
                                                     /*dilation_height=*/1,
                                                     /*dilation_width=*/1,
                                                     /*mode=*/conv_mode,
                                                     /*computeType=*/accumDataType<T>::type))) {
    state.SkipWithError(BENCHMARK_NAME " failed to cudnnSetConvolution2dDescriptor");
    return;
  }
  defer(cudnnDestroyConvolutionDescriptor(convolution_descriptor));

#ifdef CUDNN_SUPPORTS_TENSOR_OPS
  cudnnSetConvolutionMathType(convolution_descriptor, math_type);
#endif // CUDNN_SUPPORTS_TENSOR_OPS

  int out_h, out_w, out_c, out_n;

  if (PRINT_IF_ERROR(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor, input_descriptor, kernel_descriptor,
                                                           &out_n, &out_c, &out_h, &out_w))) {
    state.SkipWithError(BENCHMARK_NAME " failed to cudnnGetConvolution2dForwardOutputDim");
    return;
  }

  auto output_tensor = Tensor<T>(state,
                                 {/*batch_size=*/out_n,
                                  /*channels=*/out_c,
                                  /*image_height=*/out_h,
                                  /*image_width=*/out_w});
  if (!output_tensor.is_valid) {
    return;
  }
  cudnnTensorDescriptor_t output_descriptor = output_tensor.get();

  const auto output_image_bytes = sizeof(T) * out_n * out_c * out_h * out_w;

  size_t workspace_bytes = 0;

  cudnnConvolutionFwdAlgo_t advised_convolution_algorithm = (cudnnConvolutionFwdAlgo_t) -1;

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  for (auto _ : state) {
    cudaEventRecord(start, NULL);

    const cudnnStatus_t cudnn_err = cudnnGetConvolutionForwardAlgorithm(
        cudnn_handle, input_descriptor, kernel_descriptor, convolution_descriptor, output_descriptor,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &advised_convolution_algorithm);

    cudaEventRecord(stop, NULL);
    const auto cuda_err = cudaEventSynchronize(stop);

    state.PauseTiming();
    if (PRINT_IF_ERROR(cudnn_err)) {
      state.SkipWithError(BENCHMARK_NAME " failed to perform cudnnConvolutionForward");
      break;
    }
    if (PRINT_IF_ERROR(cuda_err)) {
      state.SkipWithError(BENCHMARK_NAME " failed to launch kernel");
      break;
    }

    float msecTotal = 0.0f;
    if (PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop))) {
      state.SkipWithError(BENCHMARK_NAME " failed to launch kernel");
      break;
    }
    state.SetIterationTime(msecTotal / 1000);
    state.ResumeTiming();
  }

  state.counters.insert({{"input_size", batch_size * channels * height * width},
                         {"input_height", height},
                         {"input_width", width},
                         {"input_channels", channels},
                         {"input_batch_size", batch_size},
                         {"output_height", out_h},
                         {"output_width", out_w},
                         {"output_channels", out_c},
                         {"output_batch_size", out_n},
                         {"filter_height", filter_height},
                         {"filter_width", filter_width},
                         {"pad_height", pad_height},
                         {"pad_width", pad_width},
                         {"stride_height", stride_height},
                         {"stride_width", stride_width},
                         {"advised_convolution_algorithm", (int) advised_convolution_algorithm},
                         {"math_type", (int) math_type}});

  state.SetItemsProcessed(int64_t(state.iterations()) * N * K * C * W * H);
}

static void LAYER_CUDNN_CONV_FORWARD_GET_ALGO_INT8(benchmark::State& state) {
  CUDNN_Impl<int8_t>(state);
}

static void LAYER_CUDNN_CONV_FORWARD_GET_ALGO_INT32(benchmark::State& state) {
  CUDNN_Impl<int32_t>(state);
}

static void LAYER_CUDNN_CONV_FORWARD_GET_ALGO_HALF(benchmark::State& state) {
  CUDNN_Impl<__half>(state);
}

static void LAYER_CUDNN_CONV_FORWARD_GET_ALGO_HALF_TENSOROP(benchmark::State& state) {
  CUDNN_Impl<__half, CUDNN_TENSOR_OP_MATH>(state);
}

static void LAYER_CUDNN_CONV_FORWARD_GET_ALGO_FLOAT(benchmark::State& state) {
  CUDNN_Impl<float>(state);
}

static void LAYER_CUDNN_CONV_FORWARD_GET_ALGO_DOUBLE(benchmark::State& state) {
  CUDNN_Impl<double>(state);
}

#define CONV_PROBLEMS INFERENCE_SERVER_CONV_PROBLEMS

BENCHMARK(LAYER_CUDNN_CONV_FORWARD_GET_ALGO_INT8)->CONV_PROBLEMS()->UseManualTime();
BENCHMARK(LAYER_CUDNN_CONV_FORWARD_GET_ALGO_INT32)->CONV_PROBLEMS()->UseManualTime();
BENCHMARK(LAYER_CUDNN_CONV_FORWARD_GET_ALGO_HALF)->CONV_PROBLEMS()->UseManualTime();
#ifdef CUDNN_SUPPORTS_TENSOR_OPS
BENCHMARK(LAYER_CUDNN_CONV_FORWARD_GET_ALGO_HALF_TENSOROP)->CONV_PROBLEMS()->UseManualTime();
#endif // CUDNN_SUPPORTS_TENSOR_OPS
BENCHMARK(LAYER_CUDNN_CONV_FORWARD_GET_ALGO_FLOAT)->CONV_PROBLEMS()->UseManualTime();
BENCHMARK(LAYER_CUDNN_CONV_FORWARD_GET_ALGO_DOUBLE)->CONV_PROBLEMS()->UseManualTime();
