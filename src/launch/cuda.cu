#include <benchmark/benchmark.h>

#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <cuda_runtime.h>

#include "init/init.hpp"
#include "utils/utils.hpp"

#include "launch/args.hpp"

enum class CUDA_LAUNCH_IMPLEMENTATION : int { EMPTY = 1, ADDTWO, RELU };

static inline std::string CUDA_LAUNCH_IMPLEMENTATION_STRING(const CUDA_LAUNCH_IMPLEMENTATION impl) {
  switch (impl) {
    case CUDA_LAUNCH_IMPLEMENTATION::EMPTY:
      return "EMPTY";
    case CUDA_LAUNCH_IMPLEMENTATION::ADDTWO:
      return "ADDTWO";
    case CUDA_LAUNCH_IMPLEMENTATION::RELU:
      return "RELU";
    default:
      return "UNDEFINED";
  }
}

template <typename T, int ITERATION_COUNT, int BLOCK_SIZE>
__global__ void cuda_empty_kernel(T *vec, size_t len) {
#pragma unroll
  for (int ii = 0; ii < ITERATION_COUNT; ii++) {
  }
}

template <typename T, int ITERATION_COUNT, int BLOCK_SIZE>
__global__ void cuda_add_two_kernel(T *vec, size_t len) {
  int index = threadIdx.x + blockIdx.x * BLOCK_SIZE;
  if (index < len) {
#pragma unroll
    for (int ii = 0; ii < ITERATION_COUNT; ii++) {
      vec[index] += 2;
    }
  }
}

template <typename T, int ITERATION_COUNT, int BLOCK_SIZE>
__global__ void cuda_relu_kernel(T *vec, size_t len) {
  int index = threadIdx.x + blockIdx.x * BLOCK_SIZE;
  const T zero{0};
  if (index < len) {
#pragma unroll
    for (int ii = 0; ii < ITERATION_COUNT; ii++) {
      vec[index] = vec[index] > zero ? vec[index] : zero;
    }
  }
}

template <CUDA_LAUNCH_IMPLEMENTATION IMPLEMENTATION, typename T, int LAUNCH_COUNT = 1, int ITERATION_COUNT = 1,
          int BLOCK_SIZE = 128>
static void CUDA_LAUNCH(benchmark::State &state) {

  const std::string IMPLEMENTATION_NAME = CUDA_LAUNCH_IMPLEMENTATION_STRING(IMPLEMENTATION);
  state.SetLabel(fmt::format("CUDA/LAUNCH/{}", IMPLEMENTATION_NAME));

  if (!has_cuda) {
    state.SkipWithError(fmt::format("CUDA/LAUNCH/{} no CUDA device found", IMPLEMENTATION_NAME).c_str());
    return;
  }

  const size_t N = state.range(0);

  const dim3 blockDim(BLOCK_SIZE);
  const dim3 gridDim(ceil(((float) N) / blockDim.x));

  if (gridDim.x >= cuda_device_prop.maxGridSize[0]) {
    const auto str = fmt::format("CUDA/LAUNCH/{} the grid dimension {} exceeds the max grid dimensions {}",
                                 IMPLEMENTATION_NAME, gridDim.x, cuda_device_prop.maxGridSize[0]);
    state.SkipWithError(str.c_str());
    return;
  }

  if (gridDim.x >= CUDA_MAX_GRID_SIZE) {
    const auto str = fmt::format("CUDA/LAUNCH/{} the grid dimension {} exceeds the max grid dimensions {}",
                                 IMPLEMENTATION_NAME, gridDim.x, CUDA_MAX_GRID_SIZE);
    state.SkipWithError(str.c_str());
    return;
  }

  auto a = std::vector<T>(N);

  std::fill(a.begin(), a.end(), 1);

  T *d_a{nullptr};

  if (PRINT_IF_ERROR(cudaMalloc((void **) &d_a, a.size() * sizeof(*a.data())))) {
    LOG(critical, "CUDA/LAUNCH/{} device memory allocation failed for vector A", IMPLEMENTATION_NAME);
    return;
  }
  defer(cudaFree(d_a));

  if (PRINT_IF_ERROR(cudaMemcpy(d_a, a.data(), a.size() * sizeof(*a.data()), cudaMemcpyHostToDevice))) {
    LOG(critical, "CUDA/LAUNCH/{} failed to copy vector to device", IMPLEMENTATION_NAME);
    return;
  }

#ifdef USE_CUDA_EVENTS
  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));
#endif // USE_CUDA_EVENTS

  for (auto _ : state) {
#ifdef USE_CUDA_EVENTS
    cudaEventRecord(start, NULL);
#endif // USE_CUDA_EVENTS

    for (int ii = 0; ii < LAUNCH_COUNT; ii++) {
      switch (IMPLEMENTATION) {
        case CUDA_LAUNCH_IMPLEMENTATION::EMPTY:
          cuda_empty_kernel<T, ITERATION_COUNT, BLOCK_SIZE><<<gridDim, blockDim>>>(d_a, N);
          break;
        case CUDA_LAUNCH_IMPLEMENTATION::ADDTWO:
          cuda_add_two_kernel<T, ITERATION_COUNT, BLOCK_SIZE><<<gridDim, blockDim>>>(d_a, N);
          break;
        case CUDA_LAUNCH_IMPLEMENTATION::RELU:
          cuda_relu_kernel<T, ITERATION_COUNT, BLOCK_SIZE><<<gridDim, blockDim>>>(d_a, N);
          break;
      }
    }

#ifdef USE_CUDA_EVENTS
    cudaEventRecord(stop, NULL);
    const auto cuda_err = cudaEventSynchronize(stop);
#else // USE_CUDA_EVENTS
    const auto cuda_err = cudaDeviceSynchronize();
#endif

    state.PauseTiming();

    if (PRINT_IF_ERROR(cuda_err)) {
      state.SkipWithError(fmt::format("CUDA/LAUNCH/{} failed to synchronize", IMPLEMENTATION_NAME).c_str());
      break;
    }
#ifdef USE_CUDA_EVENTS
    float msecTotal = 0.0f;
    if (PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop))) {
      state.SkipWithError(fmt::format("CUDA/LAUNCH/{} failed to get elapsed time", IMPLEMENTATION_NAME).c_str());
      break;
    }
    state.SetIterationTime(msecTotal / 1000);
#endif // USE_CUDA_EVENTS

    state.ResumeTiming();
  }

  state.counters.insert({{"N", N},
                         {"BLOCK_SIZE", BLOCK_SIZE},
                         {"THREAD_BLOCKS", gridDim.x},
                         {"IMPLEMENTATION_TYPE", (int) IMPLEMENTATION},
                         {"ITERATION_COUNT", ITERATION_COUNT},
                         {"LAUNCH_COUNT", LAUNCH_COUNT}});
  state.SetBytesProcessed(int64_t(state.iterations()) * ITERATION_COUNT * LAUNCH_COUNT * N);
}

template <typename T, int LAUNCH_COUNT, int ITERATION_COUNT, int BLOCK_SIZE>
static void CUDA_LAUNCH_EMPTY(benchmark::State &state) {
  return CUDA_LAUNCH<CUDA_LAUNCH_IMPLEMENTATION::EMPTY, T, LAUNCH_COUNT, ITERATION_COUNT, BLOCK_SIZE>(state);
}

template <typename T, int LAUNCH_COUNT, int ITERATION_COUNT, int BLOCK_SIZE>
static void CUDA_LAUNCH_ADDTWO(benchmark::State &state) {
  return CUDA_LAUNCH<CUDA_LAUNCH_IMPLEMENTATION::ADDTWO, T, LAUNCH_COUNT, ITERATION_COUNT, BLOCK_SIZE>(state);
}

template <typename T, int LAUNCH_COUNT, int ITERATION_COUNT, int BLOCK_SIZE>
static void CUDA_LAUNCH_RELU(benchmark::State &state) {
  return CUDA_LAUNCH<CUDA_LAUNCH_IMPLEMENTATION::RELU, T, LAUNCH_COUNT, ITERATION_COUNT, BLOCK_SIZE>(state);
}

#ifdef USE_CUDA_EVENTS
#define BENCHMARK_CUDA_LAUNCH0(B, ...) BENCHMARK_TEMPLATE(B, __VA_ARGS__)->ALL_ARGS()->UseManualTime();
#else // USE_CUDA_EVENTS
#define BENCHMARK_CUDA_LAUNCH0(B, ...) BENCHMARK_TEMPLATE(B, __VA_ARGS__)->ALL_ARGS()
#endif // USE_CUDA_EVENTS
#define BENCHMARK_CUDA_LAUNCH(B, ...)                                                                                  \
  BENCHMARK_CUDA_LAUNCH0(B, char, __VA_ARGS__);                                                                        \
  BENCHMARK_CUDA_LAUNCH0(B, int, __VA_ARGS__);                                                                         \
  BENCHMARK_CUDA_LAUNCH0(B, float, __VA_ARGS__);                                                                       \
  BENCHMARK_CUDA_LAUNCH0(B, double, __VA_ARGS__)

#define BENCHMARK_CUDA_LAUNCH_EMPTY(...) BENCHMARK_CUDA_LAUNCH(CUDA_LAUNCH_EMPTY, __VA_ARGS__)
#define BENCHMARK_CUDA_LAUNCH_ADDTWO(...) BENCHMARK_CUDA_LAUNCH(CUDA_LAUNCH_ADDTWO, __VA_ARGS__)
#define BENCHMARK_CUDA_LAUNCH_RELU(...) BENCHMARK_CUDA_LAUNCH(CUDA_LAUNCH_RELU, __VA_ARGS__)

#ifndef FAST_MODE
BENCHMARK_CUDA_LAUNCH_EMPTY(1, 1, 128);
BENCHMARK_CUDA_LAUNCH_EMPTY(4, 1, 128);
BENCHMARK_CUDA_LAUNCH_EMPTY(16, 1, 128);
BENCHMARK_CUDA_LAUNCH_EMPTY(32, 1, 128);
BENCHMARK_CUDA_LAUNCH_EMPTY(64, 1, 128);
BENCHMARK_CUDA_LAUNCH_EMPTY(128, 1, 128);
#endif // FAST_MODE
BENCHMARK_CUDA_LAUNCH_EMPTY(256, 1, 128);
BENCHMARK_CUDA_LAUNCH_EMPTY(512, 1, 128);
BENCHMARK_CUDA_LAUNCH_EMPTY(1024, 1, 128);
BENCHMARK_CUDA_LAUNCH_EMPTY(2048, 1, 128);

#ifndef FAST_MODE
BENCHMARK_CUDA_LAUNCH_ADDTWO(1, 1, 128);
BENCHMARK_CUDA_LAUNCH_ADDTWO(4, 1, 128);
BENCHMARK_CUDA_LAUNCH_ADDTWO(16, 1, 128);
BENCHMARK_CUDA_LAUNCH_ADDTWO(32, 1, 128);
BENCHMARK_CUDA_LAUNCH_ADDTWO(64, 1, 128);
BENCHMARK_CUDA_LAUNCH_ADDTWO(128, 1, 128);
#endif // FAST_MODE
BENCHMARK_CUDA_LAUNCH_ADDTWO(256, 1, 128);
BENCHMARK_CUDA_LAUNCH_ADDTWO(512, 1, 128);
BENCHMARK_CUDA_LAUNCH_ADDTWO(1024, 1, 128);
BENCHMARK_CUDA_LAUNCH_ADDTWO(2048, 1, 128);

#ifndef FAST_MODE
BENCHMARK_CUDA_LAUNCH_RELU(1, 1, 128);
BENCHMARK_CUDA_LAUNCH_RELU(4, 1, 128);
BENCHMARK_CUDA_LAUNCH_RELU(16, 1, 128);
BENCHMARK_CUDA_LAUNCH_RELU(32, 1, 128);
BENCHMARK_CUDA_LAUNCH_RELU(64, 1, 128);
BENCHMARK_CUDA_LAUNCH_RELU(128, 1, 128);
#endif // FAST_MODE
BENCHMARK_CUDA_LAUNCH_RELU(256, 1, 128);
BENCHMARK_CUDA_LAUNCH_RELU(512, 1, 128);
BENCHMARK_CUDA_LAUNCH_RELU(1024, 1, 128);
BENCHMARK_CUDA_LAUNCH_RELU(2048, 1, 128);
