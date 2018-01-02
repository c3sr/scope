#include <benchmark/benchmark.h>

#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <cuda_runtime.h>

#include "init.hpp"
#include "utils.hpp"
#include "utils_cuda.hpp"
#include "utils_vectoradd.hpp"

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

  if (!has_cuda) {
    state.SkipWithError(fmt::format("CUDA/LAUNCH/{} no CUDA device found", IMPLEMENTATION_NAME).c_str());
    return;
  }

  const size_t N = state.range(0);

  auto a = std::vector<T>(N);

  std::fill(a.begin(), a.end(), 1);

  T *d_a{nullptr};

  auto cuda_err = cudaMalloc((void **) &d_a, a.size() * sizeof(*a.data()));
  if (cuda_err != cudaSuccess) {
    LOG(critical, "CUDA/LAUNCH/{} device memory allocation failed for vector A", IMPLEMENTATION_NAME);
    return;
  }
  defer(cudaFree(d_a));

  cuda_err = CUDA_PERROR(cudaMemcpy(d_a, a.data(), a.size() * sizeof(*a.data()), cudaMemcpyHostToDevice));
  if (cuda_err != cudaSuccess) {
    return;
  }

  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim(ceil(((float) N) / blockDim.x));

#ifdef USE_CUDA_EVENTS
  cudaEvent_t start, stop;
  CUDA_PERROR(cudaEventCreate(&start));
  CUDA_PERROR(cudaEventCreate(&stop));
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
    cudaEventSynchronize(stop);
#else // USE_CUDA_EVENTS
    const auto cuda_err = cudaDeviceSynchronize();
#endif

    state.PauseTiming();
#ifdef USE_CUDA_EVENTS

    float msecTotal = 0.0f;
    if (cuda_err = CUDA_PERROR(cudaEventElapsedTime(&msecTotal, start, stop))) {
      state.SkipWithError(fmt::format("CUDA/LAUNCH/{} failed to get elapsed time", IMPLEMENTATION_NAME).c_str());
      break;
    }
    state.SetIterationTime(msecTotal / 1000);
#else  // USE_CUDA_EVENTS
    if (CUDA_PERROR(cuda_err) != cudaSuccess) {
      state.SkipWithError(fmt::format("CUDA/LAUNCH/{} failed to synchronize", IMPLEMENTATION_NAME).c_str());
      break;
    }
#endif // USE_CUDA_EVENTS

    state.ResumeTiming();
  }

  state.counters.insert({{"N", N},
                         {"BLOCK_SIZE", BLOCK_SIZE},
                         {"IMPLEMENTATION_TYPE", (int) IMPLEMENTATION},
                         {"ITERATION_COUNT", ITERATION_COUNT},
                         {"LAUNCH_COUNT", LAUNCH_COUNT}});
  state.SetBytesProcessed(int64_t(state.iterations()) * N);
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
#define BENCHMARK_CUDA_LAUNCH0(B, ...) BENCHMARK_TEMPLATE(B, __VA_ARGS__)->VECTORADD_ARGS()->UseManualTime();
#else // USE_CUDA_EVENTS
#define BENCHMARK_CUDA_LAUNCH0(B, ...) BENCHMARK_TEMPLATE(B, __VA_ARGS__)->VECTORADD_ARGS()
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

#ifndef FAST_MODE
BENCHMARK_CUDA_LAUNCH_ADDTWO(1, 1, 128);
BENCHMARK_CUDA_LAUNCH_ADDTWO(4, 1, 128);
BENCHMARK_CUDA_LAUNCH_ADDTWO(16, 1, 128);
BENCHMARK_CUDA_LAUNCH_ADDTWO(32, 1, 128);
BENCHMARK_CUDA_LAUNCH_ADDTWO(64, 1, 128);
BENCHMARK_CUDA_LAUNCH_ADDTWO(128, 1, 128);
#endif // FAST_MODE
BENCHMARK_CUDA_LAUNCH_ADDTWO(256, 1, 128);

#ifndef FAST_MODE
BENCHMARK_CUDA_LAUNCH_RELU(1, 1, 128);
BENCHMARK_CUDA_LAUNCH_RELU(4, 1, 128);
BENCHMARK_CUDA_LAUNCH_RELU(16, 1, 128);
BENCHMARK_CUDA_LAUNCH_RELU(32, 1, 128);
BENCHMARK_CUDA_LAUNCH_RELU(64, 1, 128);
BENCHMARK_CUDA_LAUNCH_RELU(128, 1, 128);
#endif // FAST_MODE
BENCHMARK_CUDA_LAUNCH_RELU(256, 1, 128);
