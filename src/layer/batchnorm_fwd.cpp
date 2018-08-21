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

// https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnBatchNormMode_t
// https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnBatchNormalizationForwardTraining
// https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnBatchNormalizationForwardInference
template <typename T, cudnnBatchNormMode_t batchnorm_mode, bool is_training>
static void LAYER_CUDNN_BATCHNORM_FWD_Impl(benchmark::State& state) {
  if (!has_cuda) {
    state.SkipWithError(BENCHMARK_NAME " no CUDA device found");
    return;
  }

  // n, c, h, w
  const auto in_n = state.range(0);
  const auto in_c = state.range(1);
  const auto in_h = state.range(2);
  const auto in_w = state.range(3);

  const float alpha = 1, beta = 0;
  const double exponential_average_factor = 1.0;  // exponentialAverageFactor
  const double epsilon                    = 1e-5; // CUDNN_BN_MIN_EPSILON

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

  DeviceMemory<T> estimated_mean_memory(state, scale_bias.data(), scale_bias_bytes);
  if (!estimated_mean_memory.is_valid) {
    return;
  }
  const auto d_estimated_mean = estimated_mean_memory.get();

  DeviceMemory<T> estimated_var_memory(state, scale_bias.data(), scale_bias_bytes);
  if (!estimated_var_memory.is_valid) {
    return;
  }
  const auto d_estimated_var = estimated_var_memory.get();

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  cudnnStatus_t cudnn_err;

  for (auto _ : state) {
    cudaEventRecord(start, NULL);
    if (is_training) {
      cudnn_err = cudnnBatchNormalizationForwardTraining(cudnn_handle,
                                                         batchnorm_mode,
                                                         &alpha,
                                                         &beta,
                                                         x_descriptor,
                                                         d_x,
                                                         x_descriptor,
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
      cudnn_err = cudnnBatchNormalizationForwardInference(cudnn_handle,
                                                          batchnorm_mode,
                                                          &alpha,
                                                          &beta,
                                                          x_descriptor,
                                                          d_x,
                                                          x_descriptor,
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
      state.SkipWithError(BENCHMARK_NAME " failed to perform cudnnBatchNormalizationForward");
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

template <typename T, cudnnBatchNormMode_t batchnorm_mode>
void LAYER_CUDNN_BATCHNORM_FWD_INFERENCE_Impl(benchmark::State& state) {
  LAYER_CUDNN_BATCHNORM_FWD_Impl<T, batchnorm_mode, false>(state);
}
template <typename T, cudnnBatchNormMode_t batchnorm_mode>
void LAYER_CUDNN_BATCHNORM_FWD_TRAINING_Impl(benchmark::State& state) {
  LAYER_CUDNN_BATCHNORM_FWD_Impl<T, batchnorm_mode, true>(state);
}

#ifdef GENERATED_BENCHMARK_LAYER

#define ENABLE_LAYER_CUDNN_BATCHNORM_FWD_INFERENCE 1
#include "generated_benchmarks.hpp"
#undef ENABLE_LAYER_CUDNN_BATCHNORM_FWD_INFERENCE

#else // GENERATED_BENCHMARK_LAYER

template <cudnnBatchNormMode_t batchnorm_mode, bool is_training>
static void LAYER_CUDNN_BATCHNORM_FWD_INT8(benchmark::State& state) {
  LAYER_CUDNN_BATCHNORM_FWD_Impl<int8_t, batchnorm_mode, is_training>(state);
}

template <cudnnBatchNormMode_t batchnorm_mode, bool is_training>
static void LAYER_CUDNN_BATCHNORM_FWD_INT32(benchmark::State& state) {
  LAYER_CUDNN_BATCHNORM_FWD_Impl<int32_t, batchnorm_mode, is_training>(state);
}

template <cudnnBatchNormMode_t batchnorm_mode, bool is_training>
static void LAYER_CUDNN_BATCHNORM_FWD_HALF(benchmark::State& state) {
  LAYER_CUDNN_BATCHNORM_FWD_Impl<__half, batchnorm_mode, is_training>(state);
}

template <cudnnBatchNormMode_t batchnorm_mode, bool is_training>
static void LAYER_CUDNN_BATCHNORM_FWD_FLOAT(benchmark::State& state) {
  LAYER_CUDNN_BATCHNORM_FWD_Impl<float, batchnorm_mode, is_training>(state);
}

template <cudnnBatchNormMode_t batchnorm_mode, bool is_training>
static void LAYER_CUDNN_BATCHNORM_FWD_DOUBLE(benchmark::State& state) {
  LAYER_CUDNN_BATCHNORM_FWD_Impl<double, batchnorm_mode, is_training>(state);
}

#define CONV_PROBLEMS INFERENCE_SERVER_CONV_PROBLEMS

#define BENCHMARK_CUDNN(b)                                                                                             \
  BENCHMARK_TEMPLATE(b, CUDNN_BATCHNORM_SPATIAL, true)->CONV_PROBLEMS()->UseManualTime();                              \
  BENCHMARK_TEMPLATE(b, CUDNN_BATCHNORM_PER_ACTIVATION, true)->CONV_PROBLEMS()->UseManualTime();                       \
  BENCHMARK_TEMPLATE(b, CUDNN_BATCHNORM_PER_ACTIVATION, false)->CONV_PROBLEMS()->UseManualTime()

/* BENCHMARK_CUDNN(LAYER_CUDNN_BATCHNORM_FWD_INT8); */
/* BENCHMARK_CUDNN(LAYER_CUDNN_BATCHNORM_FWD_INT32); */
BENCHMARK_CUDNN(LAYER_CUDNN_BATCHNORM_FWD_HALF);
BENCHMARK_CUDNN(LAYER_CUDNN_BATCHNORM_FWD_FLOAT);
// BENCHMARK_CUDNN(LAYER_CUDNN_BATCHNORM_FWD_DOUBLE);

#endif // GENERATED_BENCHMARK_LAYER
