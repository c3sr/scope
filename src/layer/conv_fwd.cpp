#define BENCHMARK_NAME "CUDNN/CONV_FWD"

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

// Calculates convolution output dimension using the definition from Caffe
static inline int calc_out_dim(int input_dim, int filter_dim, int padd, int stride) {
  return (input_dim - filter_dim + 2 * padd) / stride + 1;
}

// http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
// http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionFwdAlgo_t
template <typename T, cudnnConvolutionFwdAlgo_t convolution_algorithm, cudnnMathType_t math_type = CUDNN_DEFAULT_MATH>
static void CUDNN_Impl(benchmark::State& state) {
  if (!has_cuda) {
    state.SkipWithError(BENCHMARK_NAME " no CUDA device found");
    return;
  }
  if (math_type == CUDNN_TENSOR_OP_MATH && !detail::SupportsTensorCore(FLAG(cuda_device_id))) {
    state.SkipWithError(BENCHMARK_NAME "no Tensorcore support on current device");
    return;
  }

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

  cudnnSetConvolutionMathType(convolution_descriptor, math_type);

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
  if (cudnnGetConvolutionForwardAlgorithm(cudnn_handle, input_descriptor, kernel_descriptor, convolution_descriptor,
                                          output_descriptor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0,
                                          &advised_convolution_algorithm) != CUDNN_STATUS_SUCCESS) {
    advised_convolution_algorithm = (cudnnConvolutionFwdAlgo_t) -1;
  }

  if (std::is_same<T, int8_t>::value) {

    // Note: cudnn workspace size function doesn't work for INT8_CONFIG
    workspace_bytes = 1073741824;
  } else {
    if (PRINT_IF_ERROR(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                                                               input_descriptor,
                                                               kernel_descriptor,
                                                               convolution_descriptor,
                                                               output_descriptor,
                                                               convolution_algorithm,
                                                               &workspace_bytes))) {
      state.SkipWithError(BENCHMARK_NAME " failed to cudnnGetConvolutionForwardWorkspaceSize");
      return;
    }
  }
  // std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB" << std::endl;

  DeviceMemory<T> workspace_memory(state, workspace_bytes);
  if (!workspace_memory.is_valid) {
    return;
  }
  const auto d_workspace = workspace_memory.get();

  DeviceMemory<T> input_memory(state, input_image.data(), input_image_bytes);
  if (!input_memory.is_valid) {
    return;
  }
  const auto d_input = input_memory.get();

  DeviceMemory<T> output_memory(state, output_image_bytes);
  if (!output_memory.is_valid) {
    return;
  }
  const auto d_output = output_memory.get();

  DeviceMemory<T> kernel_memory(state, kernel.data(), kernel_bytes);
  if (!kernel_memory.is_valid) {
    return;
  }
  const auto d_kernel = kernel_memory.get();

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  for (auto _ : state) {
    cudaEventRecord(start, NULL);

    const cudnnStatus_t cudnn_err = cudnnConvolutionForward(cudnn_handle,
                                                            &alpha,
                                                            input_descriptor,
                                                            d_input,
                                                            kernel_descriptor,
                                                            d_kernel,
                                                            convolution_descriptor,
                                                            convolution_algorithm,
                                                            d_workspace,
                                                            workspace_bytes,
                                                            &beta,
                                                            output_descriptor,
                                                            d_output);

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
                         {"workspace_bytes", workspace_bytes},
                         {"workspace_megabytes", workspace_bytes / 1048576.0},
                         {"convolution_algorithm", (int) convolution_algorithm},
                         {"advised_convolution_algorithm", (int) advised_convolution_algorithm},
                         {"math_type", (int) math_type}});

  const auto P = out_h, Q = out_w;

  const auto compute_flops = [&](cudnnConvolutionFwdAlgo_t alg) {
    switch (alg) {
      case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
      case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
      case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
        // flops = 2 * filter_width * filter_height * out_w * out_h * channels * out_c * batch_size *
        // state.iterations(); 2KCRSNPQ
        return static_cast<double>(2) * static_cast<double>(K) * static_cast<double>(C) * static_cast<double>(R) *
               static_cast<double>(S) * static_cast<double>(N) * static_cast<double>(P) * static_cast<double>(Q);
      case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
      case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
        //(NCKHW + (NC +CK +NK)HW log(HW))
        return (static_cast<double>(N) * static_cast<double>(C) * static_cast<double>(K) * static_cast<double>(H) *
                static_cast<double>(W)) +
               (static_cast<double>(N) * static_cast<double>(C) + static_cast<double>(C) * static_cast<double>(K) +
                static_cast<double>(N) * static_cast<double>(K)) *
                   (static_cast<double>(H) * static_cast<double>(W)) *
                   std::log2(static_cast<double>(H) * static_cast<double>(W));
      case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
      case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
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

  state.SetItemsProcessed(int64_t(state.iterations()) * N * K * C * W * H);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_BACKWARD_INT8(benchmark::State& state) {
  CUDNN_Impl<int8_t, convolution_algorithm>(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_BACKWARD_INT32(benchmark::State& state) {
  CUDNN_Impl<int32_t, convolution_algorithm>(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_BACKWARD_HALF(benchmark::State& state) {
  CUDNN_Impl<__half, convolution_algorithm>(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_BACKWARD_HALF_TENSOROP(benchmark::State& state) {
  CUDNN_Impl<__half, convolution_algorithm, CUDNN_TENSOR_OP_MATH>(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_BACKWARD_FLOAT(benchmark::State& state) {
  CUDNN_Impl<float, convolution_algorithm>(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_BACKWARD_DOUBLE(benchmark::State& state) {
  CUDNN_Impl<double, convolution_algorithm>(state);
}

#define CONV_PROBLEMS ALL_CONV_PROBLEMS

#define BENCHMARK_CUDNN(b)                                                                                             \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0)->CONV_PROBLEMS()->UseManualTime();                          \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_BWD_DATA_ALGO_1)->CONV_PROBLEMS()->UseManualTime();                          \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT)->CONV_PROBLEMS()->UseManualTime();                        \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_BWD_DATA_ALGO_​FFT_TILING)->CONV_PROBLEMS()->UseManualTime();              \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD)->CONV_PROBLEMS()->UseManualTime();                   \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_BWD_DATA_ALGO_​WINOGRAD_NONFUSED)->CONV_PROBLEMS()->UseManualTime()

BENCHMARK_CUDNN(LAYER_CUDNN_CONV_BACKWARD_INT8);
BENCHMARK_CUDNN(LAYER_CUDNN_CONV_BACKWARD_INT32);
BENCHMARK_CUDNN(LAYER_CUDNN_CONV_BACKWARD_HALF);
BENCHMARK_CUDNN(LAYER_CUDNN_CONV_BACKWARD_HALF_TENSOROP);
BENCHMARK_CUDNN(LAYER_CUDNN_CONV_BACKWARD_FLOAT);
BENCHMARK_CUDNN(LAYER_CUDNN_CONV_BACKWARD_DOUBLE);
