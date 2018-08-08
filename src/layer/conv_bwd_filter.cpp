#define BENCHMARK_NAME "CUDNN/CONV_BWD_FILTER"

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
// https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBwdFilterAlgo_t
// https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardFilter
template <typename T, cudnnConvolutionBwdFilterAlgo_t convolution_algorithm
#ifdef CUDNN_SUPPORTS_TENSOR_OPS
          ,
          cudnnMathType_t math_type = CUDNN_DEFAULT_MATH
#endif // CUDNN_SUPPORTS_TENSOR_OPS
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
  const auto num_filters   = state.range(4);
  const auto filter_width  = state.range(5);
  const auto filter_height = state.range(6);
  const auto pad_width     = state.range(7);
  const auto pad_height    = state.range(8);
  const auto stride_width  = state.range(9);
  const auto stride_height = state.range(10);

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

  auto x_tensor = Tensor<T>(state,
                            {/*batch_size=*/batch_size,
                             /*channels=*/channels,
                             /*image_height=*/height,
                             /*image_width=*/width});
  if (!x_tensor.is_valid) {
    return;
  }
  cudnnTensorDescriptor_t x_descriptor = x_tensor.get();

  const auto dw_filter = Filter<T>(state,
                                   {/*out_channels=*/num_filters,
                                    /*in_channels=*/channels,
                                    /*kernel_height=*/filter_height,
                                    /*kernel_width=*/filter_width});
  if (!dw_filter.is_valid) {
    return;
  }
  cudnnFilterDescriptor_t dw_descriptor = dw_filter.get();

  int out_n, out_c, out_h, out_w;
  if (PRINT_IF_ERROR(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor, x_descriptor, dw_descriptor, &out_n,
                                                           &out_c, &out_h, &out_w))) {
    state.SkipWithError(BENCHMARK_NAME " failed to cudnnGetConvolution2dForwardOutputDim");
    return;
  }

  auto dy_tensor = Tensor<T>(state,
                             {/*batch_size=*/out_n,
                              /*channels=*/out_c,
                              /*image_height=*/out_h,
                              /*image_width=*/out_w});
  if (!dy_tensor.is_valid) {
    return;
  }
  cudnnTensorDescriptor_t dy_descriptor = dy_tensor.get();

  cudnnConvolutionBwdFilterAlgo_t advised_convolution_algorithm = (cudnnConvolutionBwdFilterAlgo_t) -1;
  if (IS_ERROR(cudnnGetConvolutionBackwardFilterAlgorithm(
          cudnn_handle, x_descriptor, dy_descriptor, convolution_descriptor, dw_descriptor,
          CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &advised_convolution_algorithm))) {
    advised_convolution_algorithm = (cudnnConvolutionBwdFilterAlgo_t) -1;
  }

  size_t workspace_bytes = 0;
  if (std::is_same<T, int8_t>::value) {

    // Note: cudnn workspace size function doesn't work for INT8_CONFIG
    workspace_bytes = 1073741824;
  } else {
    if (PRINT_IF_ERROR(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle,
                                                                      x_descriptor,
                                                                      dy_descriptor,
                                                                      convolution_descriptor,
                                                                      dw_descriptor,
                                                                      convolution_algorithm,
                                                                      &workspace_bytes))) {
      workspace_bytes = 1073741824;
      // state.SkipWithError(BENCHMARK_NAME " failed to cudnnGetConvolutionBackwardFilterWorkspaceSize");
      // return;
    }
  }
  // std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB" << std::endl;

  const int input_bytes  = batch_size * channels * height * width * sizeof(T);
  const int kernel_bytes = num_filters * channels * filter_height * filter_width * sizeof(T);
  auto input             = std::vector<T>(input_bytes / sizeof(T));
  auto kernel            = std::vector<T>(kernel_bytes / sizeof(T));
  std::fill(input.begin(), input.end(), detail::one<T>());
  std::fill(kernel.begin(), kernel.end(), detail::one<T>());

  const auto output_bytes = sizeof(T) * out_n * out_c * out_h * out_w;
  auto output             = std::vector<T>(output_bytes / sizeof(T));
  std::fill(output.begin(), output.end(), detail::one<T>());

  DeviceMemory<T> workspace_memory(state, workspace_bytes);
  if (!workspace_memory.is_valid) {
    return;
  }
  const auto d_workspace = workspace_memory.get();

  DeviceMemory<T> x_memory(state, input.data(), input_bytes);
  if (!x_memory.is_valid) {
    return;
  }
  const auto d_x = x_memory.get();

  DeviceMemory<T> dy_memory(state, output.data(), output_bytes);
  if (!dy_memory.is_valid) {
    return;
  }
  const auto d_dy = dy_memory.get();

  DeviceMemory<T> dw_memory(state, kernel_bytes);
  if (!dw_memory.is_valid) {
    return;
  }
  const auto d_dw = dw_memory.get();

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  for (auto _ : state) {
    cudaEventRecord(start, NULL);

    const cudnnStatus_t cudnn_err = cudnnConvolutionBackwardFilter(
        cudnn_handle, &alpha, x_descriptor, d_x, dy_descriptor, d_dy, convolution_descriptor, convolution_algorithm,
        d_workspace, workspace_bytes, &beta, dw_descriptor, d_dw);

    cudaEventRecord(stop, NULL);
    const auto cuda_err = cudaEventSynchronize(stop);

    state.PauseTiming();
    if (PRINT_IF_ERROR(cudnn_err)) {
      state.SkipWithError(BENCHMARK_NAME " failed to perform cudnnConvolutionBackwardFilter");
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
                         {"input_batch_size", batch_size},
                         {"input_channels", channels},
                         {"input_height", height},
                         {"input_width", width},
                         {"num_filters", num_filters},
                         {"filter_height", filter_height},
                         {"filter_width", filter_width},
                         {"pad_height", pad_height},
                         {"pad_width", pad_width},
                         {"stride_height", stride_height},
                         {"stride_width", stride_width},
                         {"output_size", out_n * out_c * out_h * out_w},
                         {"output_batch_size", out_n},
                         {"output_channels", out_c},
                         {"output_height", out_h},
                         {"output_width", out_w},
                         {"workspace_bytes", workspace_bytes},
                         {"workspace_megabytes", workspace_bytes / 1048576.0},
                         {"convolution_algorithm", (int) convolution_algorithm},
                         {"advised_convolution_algorithm", (int) advised_convolution_algorithm},
                         {"math_type", (int) 0}});

  const auto N = batch_size, K = num_filters, C = channels, H = height, W = width, R = filter_height, S = filter_width;
  const auto P = out_h, Q = out_w;

  const auto compute_flops = [&](cudnnConvolutionBwdFilterAlgo_t alg) {
    switch (alg) {
      case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0:
      case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1:
        // flops = 2 * filter_width * filter_height * out_w * out_h * channels * out_c * batch_size *
        // state.iterations(); 2KCRSNPQ
        return static_cast<double>(2) * static_cast<double>(K) * static_cast<double>(C) * static_cast<double>(R) *
               static_cast<double>(S) * static_cast<double>(N) * static_cast<double>(P) * static_cast<double>(Q);
      case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT:
      case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING:
        //(NCKHW + (NC +CK +NK)HW log(HW))
        return (static_cast<double>(N) * static_cast<double>(C) * static_cast<double>(K) * static_cast<double>(H) *
                static_cast<double>(W)) +
               (static_cast<double>(N) * static_cast<double>(C) + static_cast<double>(C) * static_cast<double>(K) +
                static_cast<double>(N) * static_cast<double>(K)) *
                   (static_cast<double>(H) * static_cast<double>(W)) *
                   std::log2(static_cast<double>(H) * static_cast<double>(W));
      case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED:
        return static_cast<double>(-1); // todo ... implement
      default:
        return static_cast<double>(-1);
    }
  };

  const double predicted_flops = compute_flops(convolution_algorithm);
  state.counters.insert(
      {{"predicted_flops_count", predicted_flops},
       {"predicted_flops", {predicted_flops * state.iterations(), benchmark::Counter::kAvgThreadsRate}}});

  if (advised_convolution_algorithm != -1) {
    const double predicted_advised_flops = compute_flops(advised_convolution_algorithm);
    state.counters.insert({{"predicted_advised_flops_count", predicted_advised_flops},
                           {"predicted_advised_flops",
                            {predicted_advised_flops * state.iterations(), benchmark::Counter::kAvgThreadsRate}}});
  }

  cudnnStatus_t cudnn_err;
  int max_count = 10;
  /* cudnn_err = cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnn_handle, &max_count); */
  /* if (PRINT_IF_ERROR(cudnn_err)) { */
  /*   state.SkipWithError(BENCHMARK_NAME " failed to perform cudnnGetConvolutionBackwardFilterAlgorithmMaxCount"); */
  /* } */

  cudnnConvolutionBwdFilterAlgoPerf_t perfResults[max_count];
  int returned_count;
  cudnn_err =
      cudnnFindConvolutionBackwardFilterAlgorithm(cudnn_handle, x_descriptor, dy_descriptor, convolution_descriptor,
                                                  dw_descriptor, max_count, &returned_count, perfResults);
  if (PRINT_IF_ERROR(cudnn_err)) {
    state.SkipWithError(BENCHMARK_NAME " failed to perform cudnnFindConvolutionBackwardFilterAlgorithm");
  }

  for (auto ii = 0; ii < returned_count; ii++) {
    cudnnConvolutionBwdFilterAlgoPerf_t perfResult = perfResults[ii];
    if (perfResult.algo == convolution_algorithm) {
      state.counters.insert({{"advised_time", perfResult.time},
                             {"advised_memory", perfResult.memory},
                             {"advised_determinism", (int) perfResult.determinism}});
    }
  }

  state.SetItemsProcessed(int64_t(state.iterations()) * N * K * C * W * H);
}

template <cudnnConvolutionBwdFilterAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_BWD_FILTER_INT8(benchmark::State& state) {
  CUDNN_Impl<int8_t, convolution_algorithm>(state);
}

template <cudnnConvolutionBwdFilterAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_BWD_FILTER_INT32(benchmark::State& state) {
  CUDNN_Impl<int32_t, convolution_algorithm>(state);
}

template <cudnnConvolutionBwdFilterAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_BWD_FILTER_HALF(benchmark::State& state) {
  CUDNN_Impl<__half, convolution_algorithm>(state);
}

#ifdef CUDNN_SUPPORTS_TENSOR_OPS
template <cudnnConvolutionBwdFilterAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_BWD_FILTER_HALF_TENSOROP(benchmark::State& state) {
  CUDNN_Impl<__half, convolution_algorithm, CUDNN_TENSOR_OP_MATH>(state);
}
#endif

template <cudnnConvolutionBwdFilterAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_BWD_FILTER_FLOAT(benchmark::State& state) {
  CUDNN_Impl<float, convolution_algorithm>(state);
}

template <cudnnConvolutionBwdFilterAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_BWD_FILTER_DOUBLE(benchmark::State& state) {
  CUDNN_Impl<double, convolution_algorithm>(state);
}

#define CONV_PROBLEMS ALL_INFERENCE_SERVER_CONV_PROBLEMS

#define BENCHMARK_CUDNN(b)                                                                                             \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0)->INFERENCE_SERVER_CONV_PROBLEMS()->UseManualTime();       \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1)->INFERENCE_SERVER_CONV_PROBLEMS()->UseManualTime();       \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT)->INFERENCE_SERVER_CONV_PROBLEMS()->UseManualTime();     \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3)->INFERENCE_SERVER_CONV_PROBLEMS()->UseManualTime();       \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED)                                           \
      ->INFERENCE_SERVER_CONV_PROBLEMS()                                                                               \
      ->UseManualTime();                                                                                               \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING)->INFERENCE_SERVER_CONV_PROBLEMS()->UseManualTime()

// BENCHMARK_CUDNN(LAYER_CUDNN_CONV_BWD_FILTER_INT8);
// BENCHMARK_CUDNN(LAYER_CUDNN_CONV_BWD_FILTER_INT32);
BENCHMARK_CUDNN(LAYER_CUDNN_CONV_BWD_FILTER_HALF);
#ifdef CUDNN_SUPPORTS_TENSOR_OPS
BENCHMARK_CUDNN(LAYER_CUDNN_CONV_BWD_FILTER_HALF_TENSOROP);
#endif // CUDNN_SUPPORTS_TENSOR_OPS
BENCHMARK_CUDNN(LAYER_CUDNN_CONV_BWD_FILTER_FLOAT);
BENCHMARK_CUDNN(LAYER_CUDNN_CONV_BWD_FILTER_DOUBLE);
