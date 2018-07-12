#define BENCHMARK_NAME "CUDNN/BATCHNORM_FWD"

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

// https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnBatchNormMode_t
template <typename T, cudnnBatchNormMode_t batchnorm_mode, bool is_training>
static void CUDNN_Impl(benchmark::State& state) {
  if (!has_cuda) {
    state.SkipWithError(BENCHMARK_NAME " no CUDA device found");
    return;
  }

  const float alpha = 1, beta = 0;
  const double exponential_average_factor = 1.0;  // exponentialAverageFactor
  const double epsilon                    = 1e-5; // CUDNN_BN_MIN_EPSILON

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

  const auto out_n = batch_size;
  const auto out_w = calc_conv_out_dim(width, filter_width, pad_width, stride_width);
  const auto out_h = calc_conv_out_dim(height, filter_height, pad_height, stride_height);
  const auto out_c = num_filters;

  auto x_tensor = Tensor<T>(state,
                            {/*batch_size=*/out_n,
                             /*channels=*/out_c,
                             /*image_height=*/out_h,
                             /*image_width=*/out_w});
  if (!x_tensor.is_valid) {
    return;
  }
  cudnnTensorDescriptor_t x_descriptor = x_tensor.get();

  auto y_tensor = Tensor<T>(state,
                            {/*batch_size=*/out_n,
                             /*channels=*/out_c,
                             /*image_height=*/out_h,
                             /*image_width=*/out_w});
  if (!y_tensor.is_valid) {
    return;
  }
  cudnnTensorDescriptor_t y_descriptor = y_tensor.get();

  cudnnTensorDescriptor_t scale_bias_descriptor;
  if (PRINT_IF_ERROR(cudnnDeriveBNTensorDescriptor(scale_bias_descriptor, x_descriptor, batchnorm_mode))) {
    state.SkipWithError(BENCHMARK_NAME " failed to cudnnDeriveBNTensorDescriptor");
    return;
  }

  size_t scale_bias_bytes;
  if (PRINT_IF_ERROR(cudnnGetTensorSizeInBytes(scale_bias_descriptor, &scale_bias_bytes))) {
    state.SkipWithError(BENCHMARK_NAME " failed to cudnnGetTensorSizeInBytes");
    return;
  }
  auto scale_bias = std::vector<T>(scale_bias_bytes / sizeof(T));
  std::fill(scale_bias.begin(), scale_bias.end(), detail::one<T>());

  const auto input_bytes = out_n * out_w * out_h * out_c * sizeof(T);
  auto input             = std::vector<T>(input_bytes / sizeof(T));
  std::fill(input.begin(), input.end(), detail::one<T>());

  DeviceMemory<T> x_memory(state, input.data(), input_bytes);
  if (!x_memory.is_valid) {
    return;
  }
  const auto d_x = x_memory.get();

  DeviceMemory<T> y_memory(state, input_bytes);
  if (!y_memory.is_valid) {
    return;
  }
  const auto d_y = y_memory.get();

  DeviceMemory<T> scale_memory(state, scale_bias.data(), scale_bias_bytes);
  if (!scale_memory.is_valid) {
    return;
  }
  const auto d_scale = scale_memory.get();

  DeviceMemory<T> bias_memory(state, scale_bias.data(), scale_bias_bytes);
  if (!bias_memory.is_valid) {
    return;
  }
  const auto d_bias = bias_memory.get();

  DeviceMemory<T> batch_mean_memory(state, scale_bias_bytes);
  if (!batch_mean_memory.is_valid) {
    return;
  }
  const auto d_batch_mean = batch_mean_memory.get();

  DeviceMemory<T> batch_var_memory(state, scale_bias_bytes);
  if (!batch_var_memory.is_valid) {
    return;
  }
  const auto d_batch_var = batch_var_memory.get();

  DeviceMemory<T> saved_mean_memory(state, scale_bias_bytes);
  if (!saved_mean_memory.is_valid) {
    return;
  }
  const auto d_saved_mean = saved_mean_memory.get();

  DeviceMemory<T> saved_in_var_memory(state, scale_bias_bytes);
  if (!saved_in_var_memory.is_valid) {
    return;
  }
  const auto d_saved_in_var = saved_in_var_memory.get();

  DeviceMemory<T> estimated_mean_memory(state, scale_bias_bytes);
  if (!estimated_mean_memory.is_valid) {
    return;
  }
  const auto d_estimated_mean = estimated_mean_memory.get();

  DeviceMemory<T> estimated_var_memory(state, scale_bias_bytes);
  if (!estimated_var_memory.is_valid) {
    return;
  }
  const auto d_estimated_var = estimated_var_memory.get();

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  const cudnnStatus_t cudnn_err;

  for (auto _ : state) {
    cudaEventRecord(start, NULL);
    if (is_training) {
      cudnn_err = cudnnBatchNormalizationForwardTraining(cudnn_handle,
                                                         batchnorm_mode,
                                                         &alpha,
                                                         &beta,
                                                         x_descriptor,
                                                         d_x,
                                                         y_descriptor,
                                                         d_y,
                                                         scale_bias_descriptor,
                                                         d_scale,
                                                         d_bias,
                                                         exponential_average_factor,
                                                         d_batch_mean,
                                                         d_batch_var,
                                                         epsilon,
                                                         d_saved_mean,
                                                         d_saved_in_var);
    } else {
      cudnn_err = cudnnBatchNormalizationForwardTraining(cudnn_handle,
                                                         batchnorm_mode,
                                                         &alpha,
                                                         &beta,
                                                         x_descriptor,
                                                         d_x,
                                                         y_descriptor,
                                                         d_y,
                                                         scale_bias_descriptor,
                                                         d_scale,
                                                         d_bias,
                                                         d_estimated_mean,
                                                         d_estimated_var,
                                                         epsilon);
    }

    cudaEventRecord(stop, NULL);
    state.PauseTiming();

    const auto cuda_err = cudaEventSynchronize(stop);
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
                         {"output_batch_size", out_n},
                         {"batchnorm_mode", (int) batchnorm_mode},
                         {"is_training", is_training}});
});

const auto P = out_h, Q = out_w;

const auto compute_flops = [&](cudnnBatchNormMode_t mode) {
  switch (mode) {
    case CUDNN_BATCHNORM_PER_ACTIVATION:
    case CUDNN_BATCHNORM_SPATIAL:
    case CUDNN_BATCHNORM_SPATIAL_PERSISTENT:
      return out_n * out_c * out_h * out_w;
    default:
      return static_cast<double>(-1);
  }
};

const double predicted_flops = compute_flops(batchnorm_mode);
state.counters.insert({{"predicted_flops_count", predicted_flops},
                       {"predicted_flops",
                        {predicted_flops * state.iterations(), benchmark::Counter::kAvgThreadsRate}}});

state.SetItemsProcessed(int64_t(state.iterations()) * N * K * C * W * H);
}

template <cudnnBatchNormMode_t batchnorm_mode, bool is_training>
static void LAYER_CUDNN_BATCHNORM_FORWARD_INT8(benchmark::State& state) {
  CUDNN_Impl<int8_t, batchnorm_mode>(state);
}

template <cudnnBatchNormMode_t batchnorm_mode, bool is_training>
static void LAYER_CUDNN_BATCHNORM_FORWARD_INT32(benchmark::State& state) {
  CUDNN_Impl<int32_t, batchnorm_mode>(state);
}

template <cudnnBatchNormMode_t batchnorm_mode, bool is_training>
static void LAYER_CUDNN_BATCHNORM_FORWARD_HALF(benchmark::State& state) {
  CUDNN_Impl<__half, batchnorm_mode>(state);
}

template <cudnnBatchNormMode_t batchnorm_mode, bool is_training>
static void LAYER_CUDNN_BATCHNORM_FORWARD_FLOAT(benchmark::State& state) {
  CUDNN_Impl<float, batchnorm_mode>(state);
}

template <cudnnBatchNormMode_t batchnorm_mode, bool is_training>
static void LAYER_CUDNN_BATCHNORM_FORWARD_DOUBLE(benchmark::State& state) {
  CUDNN_Impl<double, batchnorm_mode>(state);
}

#define CONV_PROBLEMS INFERENCE_SERVER_CONV_PROBLEMS

#define BENCHMARK_CUDNN(b)                                                                                             \
  BENCHMARK_TEMPLATE(b, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, true)->CONV_PROBLEMS()->UseManualTime();                   \
  BENCHMARK_TEMPLATE(b, CUDNN_BATCHNORM_SPATIAL, true)->CONV_PROBLEMS()->UseManualTime();                              \
  BENCHMARK_TEMPLATE(b, CUDNN_BATCHNORM_PER_ACTIVATION, true)->CONV_PROBLEMS()->UseManualTime();                       \
  BENCHMARK_TEMPLATE(b, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, false)->CONV_PROBLEMS()->UseManualTime();                  \
  BENCHMARK_TEMPLATE(b, CUDNN_BATCHNORM_PER_ACTIVATION, false)->CONV_PROBLEMS()->UseManualTime()

/* BENCHMARK_CUDNN(LAYER_CUDNN_BATCHNORM_FORWARD_INT8); */
/* BENCHMARK_CUDNN(LAYER_CUDNN_BATCHNORM_FORWARD_INT32); */
BENCHMARK_CUDNN(LAYER_CUDNN_BATCHNORM_FORWARD_HALF);
BENCHMARK_CUDNN(LAYER_CUDNN_BATCHNORM_FORWARD_FLOAT);
BENCHMARK_CUDNN(LAYER_CUDNN_BATCHNORM_FORWARD_DOUBLE);
