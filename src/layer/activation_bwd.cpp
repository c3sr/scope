#define BENCHMARK_NAME "CUDNN/ACTIVATION_BWD"

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

// https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnActivationMode_t
// https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnActivationBackward
template <typename T, cudnnActivationMode_t activation_mode>
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

  const auto in_n = batch_size;
  const auto in_c = num_filters;
  const auto in_h = calc_conv_out_dim(height, filter_height, pad_height, stride_height);
  const auto in_w = calc_conv_out_dim(width, filter_width, pad_width, stride_width);

  const auto out_n = in_n, out_c = in_c, out_h = in_h, out_w = in_w;

  const float alpha = 1, beta = 0;
  const double coef = 1;

  auto x_tensor = Tensor<T>(state,
                            {/*batch_size=*/in_n,
                             /*channels=*/in_c,
                             /*image_height=*/in_h,
                             /*image_width=*/in_w});
  if (!x_tensor.is_valid) {
    return;
  }
  cudnnTensorDescriptor_t x_descriptor = x_tensor.get();

  const auto input_bytes = in_n * in_c * in_w * in_h * sizeof(T);
  auto input             = std::vector<T>(input_bytes / sizeof(T));
  std::fill(input.begin(), input.end(), detail::one<T>());

  DeviceMemory<T> x_memory(state, input.data(), input_bytes);
  if (!x_memory.is_valid) {
    return;
  }
  const auto d_x = x_memory.get();

  DeviceMemory<T> dx_memory(state, input_bytes);
  if (!dx_memory.is_valid) {
    return;
  }
  const auto d_dx = dx_memory.get();

  DeviceMemory<T> y_memory(state, input.data(), input_bytes);
  if (!y_memory.is_valid) {
    return;
  }
  const auto d_y = y_memory.get();

  DeviceMemory<T> dy_memory(state, input.data(), input_bytes);
  if (!dy_memory.is_valid) {
    return;
  }
  const auto d_dy = dy_memory.get();

  cudnnActivationDescriptor_t activation_descriptor;
  if (PRINT_IF_ERROR(cudnnCreateActivationDescriptor(&activation_descriptor))) {
    state.SkipWithError(BENCHMARK_NAME " failed to cudnnCreateActivationDescriptor");
    return;
  }

  if (PRINT_IF_ERROR(
          cudnnSetActivationDescriptor(activation_descriptor, activation_mode, CUDNN_NOT_PROPAGATE_NAN, coef))) {
    state.SkipWithError(BENCHMARK_NAME " failed to cudnnSetActivationDescriptor");
    return;
  }

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  for (auto _ : state) {
    cudaEventRecord(start, NULL);

    const cudnnStatus_t cudnn_err = cudnnActivationBackward(cudnn_handle,
                                                            activation_descriptor,
                                                            &alpha,
                                                            x_descriptor,
                                                            d_y,
                                                            x_descriptor,
                                                            d_dy,
                                                            x_descriptor,
                                                            d_x,
                                                            &beta,
                                                            x_descriptor,
                                                            d_dx);

    cudaEventRecord(stop, NULL);
    const auto cuda_err = cudaEventSynchronize(stop);

    state.PauseTiming();
    if (PRINT_IF_ERROR(cudnn_err)) {
      state.SkipWithError(BENCHMARK_NAME " failed to perform cudnnActivationBackward");
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
                         {"activation_mode", (int) activation_mode}});

  const auto compute_flops = [&](cudnnActivationMode_t mode) {
    switch (mode) {
      case CUDNN_ACTIVATION_SIGMOID:
      case CUDNN_ACTIVATION_RELU:
      case CUDNN_ACTIVATION_TANH:
      case CUDNN_ACTIVATION_CLIPPED_RELU:
      case CUDNN_ACTIVATION_ELU:
      /* case CUDNN_ACTIVATION_IDENTITY: */
        return static_cast<double>(in_n * in_c * in_h * in_w);
      default:
        return static_cast<double>(-1);
    }
  };

  const double predicted_flops = compute_flops(activation_mode);
  state.counters.insert(
      {{"predicted_flops_count", predicted_flops},
       {"predicted_flops", {predicted_flops * state.iterations(), benchmark::Counter::kAvgThreadsRate}}});

  state.SetItemsProcessed(int64_t(state.iterations()) * in_n * in_c * in_h * in_w);
}

template <cudnnActivationMode_t activation_mode>
static void LAYER_CUDNN_ACTIVATION_BWD_INT8(benchmark::State& state) {
  CUDNN_Impl<int8_t, activation_mode>(state);
}

template <cudnnActivationMode_t activation_mode>
static void LAYER_CUDNN_ACTIVATION_BWD_INT32(benchmark::State& state) {
  CUDNN_Impl<int32_t, activation_mode>(state);
}

template <cudnnActivationMode_t activation_mode>
static void LAYER_CUDNN_ACTIVATION_BWD_HALF(benchmark::State& state) {
  CUDNN_Impl<__half, activation_mode>(state);
}

template <cudnnActivationMode_t activation_mode>
static void LAYER_CUDNN_ACTIVATION_BWD_FLOAT(benchmark::State& state) {
  CUDNN_Impl<float, activation_mode>(state);
}

template <cudnnActivationMode_t activation_mode>
static void LAYER_CUDNN_ACTIVATION_BWD_DOUBLE(benchmark::State& state) {
  CUDNN_Impl<double, activation_mode>(state);
}

#define CONV_PROBLEMS INFERENCE_SERVER_CONV_PROBLEMS

#define BENCHMARK_CUDNN(b)                                                                                             \
  BENCHMARK_TEMPLATE(b, CUDNN_ACTIVATION_SIGMOID)->CONV_PROBLEMS()->UseManualTime();                                   \
  BENCHMARK_TEMPLATE(b, CUDNN_ACTIVATION_RELU)->CONV_PROBLEMS()->UseManualTime();                                      \
  BENCHMARK_TEMPLATE(b, CUDNN_ACTIVATION_TANH)->CONV_PROBLEMS()->UseManualTime();                                      \
  BENCHMARK_TEMPLATE(b, CUDNN_ACTIVATION_CLIPPED_RELU)->CONV_PROBLEMS()->UseManualTime();                              \
  BENCHMARK_TEMPLATE(b, CUDNN_ACTIVATION_ELU)->CONV_PROBLEMS()->UseManualTime()                                       

/* BENCHMARK_CUDNN(LAYER_CUDNN_ACTIVATION_BWD_INT8); */
/* BENCHMARK_CUDNN(LAYER_CUDNN_ACTIVATION_BWD_INT32); */
BENCHMARK_CUDNN(LAYER_CUDNN_ACTIVATION_BWD_HALF);
BENCHMARK_CUDNN(LAYER_CUDNN_ACTIVATION_BWD_FLOAT);
BENCHMARK_CUDNN(LAYER_CUDNN_ACTIVATION_BWD_DOUBLE);
