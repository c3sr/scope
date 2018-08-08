#define BENCHMARK_NAME "CUDNN/BATCHNORM_BWD"

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
// https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnBatchNormalizationBackward
template <typename T, cudnnBatchNormMode_t batchnorm_mode>
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
  const double epsilon = 1e-5; // CUDNN_BN_MIN_EPSILON

  const auto in_n = batch_size;
  const auto in_c = num_filters;
  const auto in_h = calc_conv_out_dim(height, filter_height, pad_height, stride_height);
  const auto in_w = calc_conv_out_dim(width, filter_width, pad_width, stride_width);

  const auto out_n = in_n, out_c = in_c, out_h = in_h, out_w = in_w;

  auto x_tensor = Tensor<T>(state,
                            {/*batch_size=*/in_n,
                             /*channels=*/in_c,
                             /*image_height=*/in_h,
                             /*image_width=*/in_w});
  if (!x_tensor.is_valid) {
    return;
  }
  cudnnTensorDescriptor_t x_descriptor = x_tensor.get();

  cudnnTensorDescriptor_t scale_bias_descriptor{nullptr};
  if (PRINT_IF_ERROR(cudnnCreateTensorDescriptor(&scale_bias_descriptor))) {
    state.SkipWithError(BENCHMARK_NAME " failed to cudnnCreateTensorDescriptor");
    return;
  }

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

  const auto input_bytes = in_n * in_c * in_w * in_h * sizeof(T);
  auto input             = std::vector<T>(input_bytes / sizeof(T));
  std::fill(input.begin(), input.end(), detail::one<T>());

  DeviceMemory<T> x_memory(state, input.data(), input_bytes);
  if (!x_memory.is_valid) {
    return;
  }
  const auto d_x = x_memory.get();

  DeviceMemory<T> dy_memory(state, input.data(), input_bytes);
  if (!dy_memory.is_valid) {
    return;
  }
  const auto d_dy = dy_memory.get();

  DeviceMemory<T> dx_memory(state, input_bytes);
  if (!dx_memory.is_valid) {
    return;
  }
  const auto d_dx = dx_memory.get();

  DeviceMemory<T> scale_memory(state, scale_bias.data(), scale_bias_bytes);
  if (!scale_memory.is_valid) {
    return;
  }
  const auto d_scale = scale_memory.get();

  DeviceMemory<T> dscale_memory(state, scale_bias_bytes);
  if (!dscale_memory.is_valid) {
    return;
  }
  const auto d_dscale = dscale_memory.get();

  DeviceMemory<T> dbias_memory(state, scale_bias_bytes);
  if (!dbias_memory.is_valid) {
    return;
  }
  const auto d_dbias = dbias_memory.get();

  DeviceMemory<T> saved_mean_memory(state, scale_bias.data(), scale_bias_bytes);
  if (!saved_mean_memory.is_valid) {
    return;
  }
  const auto d_saved_mean = saved_mean_memory.get();

  DeviceMemory<T> saved_in_var_memory(state, scale_bias.data(), scale_bias_bytes);
  if (!saved_in_var_memory.is_valid) {
    return;
  }
  const auto d_saved_in_var = saved_in_var_memory.get();

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  cudnnStatus_t cudnn_err;

  for (auto _ : state) {
    cudaEventRecord(start, NULL);
    cudnn_err = cudnnBatchNormalizationBackward(
        cudnn_handle, batchnorm_mode, &alpha, &beta, &alpha, &beta, x_descriptor, d_x, x_descriptor, d_dy, x_descriptor,
        d_dx, scale_bias_descriptor, d_scale, d_dscale, d_dbias, epsilon, d_saved_mean, d_saved_in_var);

    cudaEventRecord(stop, NULL);
    state.PauseTiming();

    const auto cuda_err = cudaEventSynchronize(stop);
    if (PRINT_IF_ERROR(cudnn_err)) {
      state.SkipWithError(BENCHMARK_NAME " failed to perform cudnnBatchNormalizationBackward");
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
                         {"batchnorm_mode", (int) batchnorm_mode}});

  const auto compute_flops = [&](cudnnBatchNormMode_t mode) {
    switch (mode) {
      case CUDNN_BATCHNORM_PER_ACTIVATION:
      case CUDNN_BATCHNORM_SPATIAL:
      /* case CUDNN_BATCHNORM_SPATIAL_PERSISTENT: */
        return static_cast<double>(in_n * in_c * in_h * in_w);
      default:
        return static_cast<double>(-1);
    }
  };

  const double predicted_flops = compute_flops(batchnorm_mode);
  state.counters.insert(
      {{"predicted_flops_count", predicted_flops},
       {"predicted_flops", {predicted_flops * state.iterations(), benchmark::Counter::kAvgThreadsRate}}});

  state.SetItemsProcessed(int64_t(state.iterations()) * in_n * in_c * in_h * in_w);
}

template <cudnnBatchNormMode_t batchnorm_mode>
static void LAYER_CUDNN_BATCHNORM_BWD_INT8(benchmark::State& state) {
  CUDNN_Impl<int8_t, batchnorm_mode>(state);
}

template <cudnnBatchNormMode_t batchnorm_mode>
static void LAYER_CUDNN_BATCHNORM_BWD_INT32(benchmark::State& state) {
  CUDNN_Impl<int32_t, batchnorm_mode>(state);
}

template <cudnnBatchNormMode_t batchnorm_mode>
static void LAYER_CUDNN_BATCHNORM_BWD_HALF(benchmark::State& state) {
  CUDNN_Impl<__half, batchnorm_mode>(state);
}

template <cudnnBatchNormMode_t batchnorm_mode>
static void LAYER_CUDNN_BATCHNORM_BWD_FLOAT(benchmark::State& state) {
  CUDNN_Impl<float, batchnorm_mode>(state);
}

template <cudnnBatchNormMode_t batchnorm_mode>
static void LAYER_CUDNN_BATCHNORM_BWD_DOUBLE(benchmark::State& state) {
  CUDNN_Impl<double, batchnorm_mode>(state);
}

#define CONV_PROBLEMS INFERENCE_SERVER_CONV_PROBLEMS

#define BENCHMARK_CUDNN(b)                                                                                             \
  BENCHMARK_TEMPLATE(b, CUDNN_BATCHNORM_SPATIAL)->CONV_PROBLEMS()->UseManualTime();                                    \
  BENCHMARK_TEMPLATE(b, CUDNN_BATCHNORM_PER_ACTIVATION)->CONV_PROBLEMS()->UseManualTime();

/* BENCHMARK_CUDNN(LAYER_CUDNN_BATCHNORM_BWD_INT8); */
/* BENCHMARK_CUDNN(LAYER_CUDNN_BATCHNORM_BWD_INT32); */
BENCHMARK_CUDNN(LAYER_CUDNN_BATCHNORM_BWD_HALF);
BENCHMARK_CUDNN(LAYER_CUDNN_BATCHNORM_BWD_FLOAT);
BENCHMARK_CUDNN(LAYER_CUDNN_BATCHNORM_BWD_DOUBLE);
