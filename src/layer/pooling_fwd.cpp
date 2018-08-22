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

// https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnPoolingForward
// https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnGetPooling2dForwardOutputDim
// https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnSetPooling2dDescriptor
template <typename T, cudnnPoolingMode_t pooling_mode>
static void LAYER_CUDNN_POOLING_FWD_Impl(benchmark::State& state) {
  if (!has_cuda) {
    state.SkipWithError(BENCHMARK_NAME " no CUDA device found");
    return;
  }

  const auto in_n         = state.range(0);
  const auto in_c         = state.range(1);
  const auto in_h         = state.range(2);
  const auto in_w         = state.range(3);
  const auto win_h        = state.range(4);
  const auto win_w        = state.range(5);
  const auto vert_padding = state.range(6);
  const auto hori_padding = state.range(7);
  const auto vert_stride  = state.range(8);
  const auto hori_stride  = state.range(9);

  const float alpha = 1, beta = 0;

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

  if (PRINT_IF_ERROR(cudnnSetPooling2dDescriptor(pooling_descriptor,
                                                 pooling_mode,
                                                 CUDNN_NOT_PROPAGATE_NAN,
                                                 win_h,
                                                 win_w,
                                                 vert_padding,
                                                 hori_padding,
                                                 vert_stride,
                                                 hori_stride))) {
    state.SkipWithError(BENCHMARK_NAME " failed to cudnnSetPooling2dDescriptor");
    return;
  }
  defer(cudnnDestroyPoolingDescriptor(pooling_descriptor));

  int out_n, out_c, out_h, out_w;
  if (PRINT_IF_ERROR(
          cudnnGetPooling2dForwardOutputDim(pooling_descriptor, x_descriptor, &out_n, &out_c, &out_h, &out_w))) {
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
      state.SkipWithError(BENCHMARK_NAME " failed to perform cudnnPoolingForward");
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

  state.counters.insert({{"input_size", in_n * in_c * in_h * in_w},
                         {"input_batch_size", in_n},
                         {"input_channels", in_c},
                         {"input_height", in_h},
                         {"input_width", in_w},
                         {"output_size", out_n * out_c * out_h * out_w},
                         {"output_batch_size", out_n},
                         {"output_channels", out_c},
                         {"output_height", out_h},
                         {"output_width", out_w},
                         {"window_height", win_h},
                         {"window_width", win_w},
                         {"vertical_padding", vert_padding},
                         {"horizontal_padding", hori_padding},
                         {"vertical_stride", vert_stride},
                         {"horizontal_stride", hori_stride},
                         {"pooling_mode", (int) pooling_mode}});

  const double predicted_flops = in_n * in_c * in_h * in_w;
  state.counters.insert(
      {{"predicted_flops_count", predicted_flops},
       {"predicted_flops", {predicted_flops * state.iterations(), benchmark::Counter::kAvgThreadsRate}}});

  state.SetItemsProcessed(int64_t(state.iterations()) * in_n * in_c * in_h * in_w);
}

#ifdef GENERATED_BENCHMARK_LAYER

#define ENABLE_LAYER_CUDNN_POOLING_FWD 1
#include "generated_benchmarks.hpp"
#undef ENABLE_LAYER_CUDNN_POOLING_FWD

#else // GENERATED_BENCHMARK_LAYER

template <cudnnPoolingMode_t pooling_mode>
static void LAYER_CUDNN_POOLING_FWD_INT8(benchmark::State& state) {
  LAYER_CUDNN_POOLING_FWD_Impl<int8_t, pooling_mode>(state);
}

template <cudnnPoolingMode_t pooling_mode>
static void LAYER_CUDNN_POOLING_FWD_INT32(benchmark::State& state) {
  LAYER_CUDNN_POOLING_FWD_Impl<int32_t, pooling_mode>(state);
}

template <cudnnPoolingMode_t pooling_mode>
static void LAYER_CUDNN_POOLING_FWD_HALF(benchmark::State& state) {
  LAYER_CUDNN_POOLING_FWD_Impl<__half, pooling_mode>(state);
}

template <cudnnPoolingMode_t pooling_mode>
static void LAYER_CUDNN_POOLING_FWD_FLOAT(benchmark::State& state) {
  LAYER_CUDNN_POOLING_FWD_Impl<float, pooling_mode>(state);
}

template <cudnnPoolingMode_t pooling_mode>
static void LAYER_CUDNN_POOLING_FWD_DOUBLE(benchmark::State& state) {
  LAYER_CUDNN_POOLING_FWD_Impl<double, pooling_mode>(state);
}

#define CONV_PROBLEMS INFERENCE_SERVER_CONV_PROBLEMS

#define BENCHMARK_CUDNN(b)                                                                                             \
  BENCHMARK_TEMPLATE(b, CUDNN_POOLING_MAX)->CONV_PROBLEMS()->UseManualTime();                                          \
  BENCHMARK_TEMPLATE(b, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING)->CONV_PROBLEMS()->UseManualTime();                \
  BENCHMARK_TEMPLATE(b, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING)->CONV_PROBLEMS()->UseManualTime();                \
  BENCHMARK_TEMPLATE(b, CUDNN_POOLING_MAX_DETERMINISTIC)->CONV_PROBLEMS()->UseManualTime();

/* BENCHMARK_CUDNN(LAYER_CUDNN_POOLING_FWD_INT8); */
/* BENCHMARK_CUDNN(LAYER_CUDNN_POOLING_FWD_INT32); */
BENCHMARK_CUDNN(LAYER_CUDNN_POOLING_FWD_HALF);
BENCHMARK_CUDNN(LAYER_CUDNN_POOLING_FWD_FLOAT);
// BENCHMARK_CUDNN(LAYER_CUDNN_POOLING_FWD_DOUBLE);

#endif // GENERATED_BENCHMARK_LAYER
