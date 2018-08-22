#define BENCHMARK_NAME "CUDNN/SOFTMAX_FWD"

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

// https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnSoftmaxMode_t
// https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnSoftmaxForward
template <typename T, cudnnSoftmaxAlgorithm_t softmax_algorithm, cudnnSoftmaxMode_t softmax_mode>
static void LAYER_CUDNN_SOFTMAX_FWD_Impl(benchmark::State& state) {
  if (!has_cuda) {
    state.SkipWithError(BENCHMARK_NAME " no CUDA device found");
    return;
  }

  const auto in_n = state.range(0);
  const auto in_c = state.range(1);
  const auto in_h = state.range(2) == -1 ? 1 : state.range(2);
  const auto in_w = state.range(3) == -1 ? 1 : state.range(3);

  const float alpha = 1, beta = 0;

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

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  cudnnStatus_t cudnn_err;

  for (auto _ : state) {
    cudaEventRecord(start, NULL);
    cudnn_err = cudnnSoftmaxForward(
        cudnn_handle, softmax_algorithm, softmax_mode, &alpha, x_descriptor, d_x, &beta, x_descriptor, d_y);
    cudaEventRecord(stop, NULL);
    state.PauseTiming();

    const auto cuda_err = cudaEventSynchronize(stop);
    if (PRINT_IF_ERROR(cudnn_err)) {
      state.SkipWithError(BENCHMARK_NAME " failed to perform cudnnSoftmaxForward");
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
                         {"softmax_algorithm", (int) softmax_algorithm},
                         {"softmax_mode", (int) softmax_mode}});

  const auto compute_flops = [&](cudnnSoftmaxMode_t mode) {
    switch (mode) {
      case CUDNN_SOFTMAX_MODE_INSTANCE:
      case CUDNN_SOFTMAX_MODE_CHANNEL:
        return static_cast<double>(in_n * in_c * in_h * in_w);
      default:
        return static_cast<double>(-1);
    }
  };

  const double predicted_flops = compute_flops(softmax_mode);
  state.counters.insert(
      {{"predicted_flops_count", predicted_flops},
       {"predicted_flops", {predicted_flops * state.iterations(), benchmark::Counter::kAvgThreadsRate}}});

  state.SetItemsProcessed(int64_t(state.iterations()) * in_n * in_c * in_h * in_w);
}

#ifdef GENERATED_BENCHMARK_LAYER

#define ENABLE_LAYER_CUDNN_SOFTMAX_FWD 1
#include "generated_benchmarks.hpp"
#undef ENABLE_LAYER_CUDNN_SOFTMAX_FWD

#else // GENERATED_BENCHMARK_LAYER

template <cudnnSoftmaxAlgorithm_t softmax_algorithm, cudnnSoftmaxMode_t softmax_mode>
static void LAYER_CUDNN_SOFTMAX_FWD_INT8(benchmark::State& state) {
  LAYER_CUDNN_SOFTMAX_FWD_Impl<int8_t, softmax_algorithm, softmax_mode>(state);
}

template <cudnnSoftmaxAlgorithm_t softmax_algorithm, cudnnSoftmaxMode_t softmax_mode>
static void LAYER_CUDNN_SOFTMAX_FWD_INT32(benchmark::State& state) {
  LAYER_CUDNN_SOFTMAX_FWD_Impl<int32_t, softmax_algorithm, softmax_mode>(state);
}

template <cudnnSoftmaxAlgorithm_t softmax_algorithm, cudnnSoftmaxMode_t softmax_mode>
static void LAYER_CUDNN_SOFTMAX_FWD_HALF(benchmark::State& state) {
  LAYER_CUDNN_SOFTMAX_FWD_Impl<__half, softmax_algorithm, softmax_mode>(state);
}

template <cudnnSoftmaxAlgorithm_t softmax_algorithm, cudnnSoftmaxMode_t softmax_mode>
static void LAYER_CUDNN_SOFTMAX_FWD_FLOAT(benchmark::State& state) {
  LAYER_CUDNN_SOFTMAX_FWD_Impl<float, softmax_algorithm, softmax_mode>(state);
}

template <cudnnSoftmaxAlgorithm_t softmax_algorithm, cudnnSoftmaxMode_t softmax_mode>
static void LAYER_CUDNN_SOFTMAX_FWD_DOUBLE(benchmark::State& state) {
  LAYER_CUDNN_SOFTMAX_FWD_Impl<double, softmax_algorithm, softmax_mode>(state);
}

#define CONV_PROBLEMS INFERENCE_SERVER_CONV_PROBLEMS

#define BENCHMARK_CUDNN0(b, SOFTMAX_MODE)                                                                              \
  BENCHMARK_TEMPLATE(b, CUDNN_SOFTMAX_FAST, SOFTMAX_MODE)->CONV_PROBLEMS()->UseManualTime();                           \
  BENCHMARK_TEMPLATE(b, CUDNN_SOFTMAX_ACCURATE, SOFTMAX_MODE)->CONV_PROBLEMS()->UseManualTime();                       \
  BENCHMARK_TEMPLATE(b, CUDNN_SOFTMAX_LOG, SOFTMAX_MODE)->CONV_PROBLEMS()->UseManualTime()

#define BENCHMARK_CUDNN(b)                                                                                             \
  BENCHMARK_CUDNN0(b, CUDNN_SOFTMAX_MODE_INSTANCE);                                                                    \
  BENCHMARK_CUDNN0(b, CUDNN_SOFTMAX_MODE_CHANNEL)

/* BENCHMARK_CUDNN(LAYER_CUDNN_SOFTMAX_FWD_INT8); */
/* BENCHMARK_CUDNN(LAYER_CUDNN_SOFTMAX_FWD_INT32); */
BENCHMARK_CUDNN(LAYER_CUDNN_SOFTMAX_FWD_HALF);
BENCHMARK_CUDNN(LAYER_CUDNN_SOFTMAX_FWD_FLOAT);
// BENCHMARK_CUDNN(LAYER_CUDNN_SOFTMAX_FWD_DOUBLE);

#endif // GENERATED_BENCHMARK_LAYER
