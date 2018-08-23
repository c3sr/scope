#define BENCHMARK_NAME "CUDNN/DROPOUT_FWD"

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

// https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnDropoutForward
// https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnDropoutGetReserveSpaceSize
template <typename T>
static void LAYER_CUDNN_DROPOUT_FWD_Impl(benchmark::State& state) {
  if (!has_cuda) {
    state.SkipWithError(BENCHMARK_NAME " no CUDA device found");
    return;
  }

  const auto in_n = state.range(0);
  const auto in_c = state.range(1);
  const auto in_h = state.range(2) == -1 ? 1 : state.range(2);
  const auto in_w = state.range(3) == -1 ? 1 : state.range(3);

  const float dropout = 0.5;
  const uint64_t seed = 0;

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

  cudnnDropoutDescriptor_t dropout_descriptor;
  if (PRINT_IF_ERROR(cudnnCreateDropoutDescriptor(&dropout_descriptor))) {
    state.SkipWithError(BENCHMARK_NAME " failed to cudnnCreateDropoutDescriptor");
    return;
  }

  size_t states_bytes = 0;
  if (PRINT_IF_ERROR(cudnnDropoutGetStatesSize(cudnn_handle, &states_bytes))) {
    state.SkipWithError(BENCHMARK_NAME " failed to cudnnDropoutGetStatesSize");
    return;
  }

  DeviceMemory<T> states_memory(state, states_bytes);
  if (!states_memory.is_valid) {
    return;
  }
  const auto d_states = states_memory.get();

  if (PRINT_IF_ERROR(
          cudnnSetDropoutDescriptor(dropout_descriptor, cudnn_handle, dropout, d_states, states_bytes, seed))) {
    state.SkipWithError(BENCHMARK_NAME " failed to cudnnSetDropoutDescriptor");
    return;
  }
  defer(cudnnDestroyDropoutDescriptor(dropout_descriptor));

  size_t reserve_space_bytes = 0;
  if (PRINT_IF_ERROR(cudnnDropoutGetReserveSpaceSize(x_descriptor, &reserve_space_bytes))) {
    state.SkipWithError(BENCHMARK_NAME " failed to cudnnDropoutGetStatesSize");
    return;
  }

  DeviceMemory<T> reserve_space_memory(state, reserve_space_bytes);
  if (!reserve_space_memory.is_valid) {
    return;
  }
  const auto d_reserve_space = reserve_space_memory.get();

  const auto input_bytes = in_n * in_w * in_h * in_c * sizeof(T);
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

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  for (auto _ : state) {
    cudaEventRecord(start, NULL);

    const cudnnStatus_t cudnn_err = cudnnDropoutForward(
        cudnn_handle, dropout_descriptor, x_descriptor, d_x, x_descriptor, d_y, d_reserve_space, reserve_space_bytes);

    cudaEventRecord(stop, NULL);
    const auto cuda_err = cudaEventSynchronize(stop);

    state.PauseTiming();
    if (PRINT_IF_ERROR(cudnn_err)) {
      state.SkipWithError(BENCHMARK_NAME " failed to perform cudnnDropoutForward");
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
                         {"dropout", dropout}});

  const double predicted_flops = in_n * in_c * in_h * in_w;
  state.counters.insert(
      {{"predicted_flops_count", predicted_flops},
       {"predicted_flops", {predicted_flops * state.iterations(), benchmark::Counter::kAvgThreadsRate}}});

  state.SetItemsProcessed(int64_t(state.iterations()) * in_n * in_c * in_h * in_w);
}

#ifdef GENERATED_BENCHMARK_LAYER

#define ENABLE_LAYER_CUDNN_DROPOUT_FWD 1
#include "generated_benchmarks.hpp"
#undef ENABLE_LAYER_CUDNN_DROPOUT_FWD

#else // GENERATED_BENCHMARK_LAYER

static void LAYER_CUDNN_DROPOUT_FWD_INT8(benchmark::State& state) {
  LAYER_CUDNN_DROPOUT_FWD_Impl<int8_t>(state);
}

static void LAYER_CUDNN_DROPOUT_FWD_INT32(benchmark::State& state) {
  LAYER_CUDNN_DROPOUT_FWD_Impl<int32_t>(state);
}

static void LAYER_CUDNN_DROPOUT_FWD_HALF(benchmark::State& state) {
  LAYER_CUDNN_DROPOUT_FWD_Impl<__half>(state);
}

static void LAYER_CUDNN_DROPOUT_FWD_FLOAT(benchmark::State& state) {
  LAYER_CUDNN_DROPOUT_FWD_Impl<float>(state);
}

static void LAYER_CUDNN_DROPOUT_FWD_DOUBLE(benchmark::State& state) {
  LAYER_CUDNN_DROPOUT_FWD_Impl<double>(state);
}

#define CONV_PROBLEMS INFERENCE_SERVER_CONV_PROBLEMS

/* BENCHMARK(LAYER_CUDNN_DROPOUT_FWD_INT8)->INFERENCE_SERVER_CONV_PROBLEMS()->UseManualTime(); */
/* BENCHMARK(LAYER_CUDNN_DROPOUT_FWD_INT32)->INFERENCE_SERVER_CONV_PROBLEMS()->UseManualTime(); */
BENCHMARK(LAYER_CUDNN_DROPOUT_FWD_HALF)->INFERENCE_SERVER_CONV_PROBLEMS()->UseManualTime();
BENCHMARK(LAYER_CUDNN_DROPOUT_FWD_FLOAT)->INFERENCE_SERVER_CONV_PROBLEMS()->UseManualTime();
// BENCHMARK(LAYER_CUDNN_DROPOUT_FWD_DOUBLE)->INFERENCE_SERVER_CONV_PROBLEMS()->UseManualTime();

#endif // GENERATED_BENCHMARK_LAYER
