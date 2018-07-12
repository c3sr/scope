#define BENCHMARK_NAME "CUDNN/POOLING_FWD"

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

// calculates convolution output dimension
static inline int calc_conv_out_dim(int input_dim, int filter_dim, int padd, int stride) {
  return (input_dim - filter_dim + 2 * padd) / stride + 1;
}

// https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnPoolingForward
// https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnGetPooling2dForwardOutputDim
template <typename T>
static void CUDNN_Impl(benchmark::State& state) {
  if (!has_cuda) {
    state.SkipWithError(BENCHMARK_NAME " no CUDA device found");
    return;
  }

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

  const float alpha = 1, beta = 0;

  const auto in_n = batch_size;
  const auto in_w = calc_conv_out_dim(width, filter_width, pad_width, stride_width);
  const auto in_h = calc_conv_out_dim(height, filter_height, pad_height, stride_height);
  const auto in_c = num_filters;

  auto x_tensor = Tensor<T>(state,
                            {/*batch_size=*/in_n,
                             /*channels=*/in_c,
                             /*image_height=*/in_h,
                             /*image_width=*/in_w});
  if (!x_tensor.is_valid) {
    return;
  }
  cudnnTensorDescriptor_t x_descriptor = x_tensor.get();

  cudnnPoolingDescriptor_t pooling_descriptor;
  if (PRINT_IF_ERROR(cudnnCreatePoolingDescriptor(&pooling_descriptor))) {
    state.SkipWithError(BENCHMARK_NAME " failed to cudnnCreatePoolingDescriptor");
    return;
  }

  int out_n, out_c, out_h, out_w;
  if (PRINT_IF_ERROR(
          cudnnGetPooling2dForwardOutputDim(&pooling_descriptor, x_descriptor, &out_n, &out_c, &out_h, &out_w))) {
    state.SkipWithError(BENCHMARK_NAME " failed to cudnnGetPooling2dForwardOutputDim");
    return;
  }

  auto y_tensor = Tensor<T>(state,
                            {/*batch_size=*/out_n,
                             /*channels=*/out_c,
                             /*image_height=*/out_h,
                             /*image_width=*/out_w});
  if (!y_tensor.is_valid) {
    return;
  }
  cudnnTensorDescriptor_t y_descriptor = y_tensor.get();

  const auto input_bytes = in_n * in_w * in_h * in_c * sizeof(T);
  auto input             = std::vector<T>(input_bytes / sizeof(T));
  std::fill(input.begin(), input.end(), detail::one<T>());

  const auto output_bytes = out_n * out_w * out_h * out_c * sizeof(T);

  DeviceMemory<T> x_memory(state, input.data(), input_bytes);
  if (!x_memory.is_valid) {
    return;
  }
  const auto d_x = x_memory.get();

  DeviceMemory<T> y_memory(state, output_bytes);
  if (!y_memory.is_valid) {
    return;
  }
  const auto d_y = y_memory.get();

  if (PRINT_IF_ERROR(
          cudnnSetActivationDescriptor(pooling_descriptor, activation_mode, CUDNN_NOT_PROPAGATE_NAN, coef))) {
    state.SkipWithError(BENCHMARK_NAME " failed to cudnnSetActivationDescriptor");
    return;
  }
  defer(cudnnDestroyPoolingDescriptor(pooling_descriptor));

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  for (auto _ : state) {
    cudaEventRecord(start, NULL);

    const cudnnStatus_t cudnn_err =
        cudnnPoolingForward(cudnn_handle, pooling_descriptor, &alpha, x_descriptor, d_x, &beta, y_descriptor, d_y);

    cudaEventRecord(stop, NULL);
    const auto cuda_err = cudaEventSynchronize(stop);

    state.PauseTiming();
    if (PRINT_IF_ERROR(cudnn_err)) {
      state.SkipWithError(BENCHMARK_NAME " failed to perform cudnnActivationForward");
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
                         {"num_filters", num_filters},
                         {"filter_height", filter_height},
                         {"filter_width", filter_width},
                         {"pad_height", pad_height},
                         {"pad_width", pad_width},
                         {"stride_height", stride_height},
                         {"stride_width", stride_width},
                         {"output_size", out_n * out_c * out_h * out_w},
                         {"output_height", out_h},
                         {"output_width", out_w},
                         {"output_channels", out_c},
                         {"output_batch_size", out_n}});

  const double predicted_flops = in_n * in_c * in_h * in_w;
  state.counters.insert(
      {{"predicted_flops_count", predicted_flops},
       {"predicted_flops", {predicted_flops * state.iterations(), benchmark::Counter::kAvgThreadsRate}}});

  state.SetItemsProcessed(int64_t(state.iterations()) * N * K * C * W * H);
}

static void LAYER_CUDNN_POOLING_FORWARD_INT8(benchmark::State& state) {
  CUDNN_Impl<int8_t, activation_mode>(state);
}

static void LAYER_CUDNN_POOLING_FORWARD_INT32(benchmark::State& state) {
  CUDNN_Impl<int32_t, activation_mode>(state);
}

static void LAYER_CUDNN_POOLING_FORWARD_HALF(benchmark::State& state) {
  CUDNN_Impl<__half, activation_mode>(state);
}

static void LAYER_CUDNN_POOLING_FORWARD_FLOAT(benchmark::State& state) {
  CUDNN_Impl<float, activation_mode>(state);
}

static void LAYER_CUDNN_POOLING_FORWARD_DOUBLE(benchmark::State& state) {
  CUDNN_Impl<double, activation_mode>(state);
}

#define CONV_PROBLEMS INFERENCE_SERVER_CONV_PROBLEMS

BENCHMARK(LAYER_CUDNN_POOLING_FORWARD_INT8)->INFERENCE_SERVER_CONV_PROBLEMS()->UseManualTime();
BENCHMARK(LAYER_CUDNN_POOLING_FORWARD_INT32)->INFERENCE_SERVER_CONV_PROBLEMS()->UseManualTime();
BENCHMARK(LAYER_CUDNN_POOLING_FORWARD_HALF)->INFERENCE_SERVER_CONV_PROBLEMS()->UseManualTime();
BENCHMARK(LAYER_CUDNN_POOLING_FORWARD_FLOAT)->INFERENCE_SERVER_CONV_PROBLEMS()->UseManualTime();
BENCHMARK(LAYER_CUDNN_POOLING_FORWARD_DOUBLE)->INFERENCE_SERVER_CONV_PROBLEMS()->UseManualTime();
