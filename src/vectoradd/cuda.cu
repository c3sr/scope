
#include <benchmark/benchmark.h>

#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <cuda_runtime.h>

#include "init/init.hpp"
#include "utils/utils.hpp"
#include "vectoradd/args.hpp"

template <typename T, int COARSINING_FACTOR = 1, int BLOCK_SIZE = 1>
__global__ void cuda_vector_add(T *in1, T *in2, T *out, size_t len) {
  // todo: implement COARSINING_FACTOR
  int index = threadIdx.x + blockIdx.x * BLOCK_SIZE;
  if (index < len) {
    out[index] = in1[index] + in2[index];
  }
}

template <typename T, int COARSINING_FACTOR = 1, int BLOCK_SIZE = 128>
static void CUDA_VECTOR_ADD(benchmark::State &state) {
  if (!has_cuda) {
    state.SkipWithError("CUDA/VECTOR_ADD/BASIC no CUDA device found");
    return;
  }

  const size_t N = state.range(0);

  const dim3 blockDim(BLOCK_SIZE);
  const dim3 gridDim(ceil(((float) N) / blockDim.x));

  if (gridDim.x >= cuda_device_prop.maxGridSize[0]) {
    const auto str = fmt::format("CUDA/VECTOR_ADD/BASIC the grid dimension {} exceeds the max grid dimensions {}",
                                 gridDim.x, cuda_device_prop.maxGridSize[0]);
    state.SkipWithError(str.c_str());
    return;
  }

  if (gridDim.x >= CUDA_MAX_GRID_SIZE) {
    const auto str = fmt::format("CUDA/VECTOR_ADD/BASIC the grid dimension {} exceeds the max grid dimensions {}",
                                 gridDim.x, CUDA_MAX_GRID_SIZE);
    state.SkipWithError(str.c_str());
    return;
  }

  auto a = std::vector<T>(N);
  auto b = std::vector<T>(N);
  auto c = std::vector<T>(N);

  std::fill(a.begin(), a.end(), 1);
  std::fill(b.begin(), b.end(), 1);
  std::fill(c.begin(), c.end(), 0);

  T *d_a{nullptr}, *d_b{nullptr}, *d_c{nullptr};

  if (PRINT_IF_ERROR(cudaMalloc((void **) &d_a, a.size() * sizeof(*a.data())))) {
    LOG(critical, "CUDA/VECTOR_ADD/BASIC device memory allocation failed for vector A");
    state.SkipWithError("CUDA/VECTOR_ADD/BASIC device memory allocation failed for vector A");
    return;
  }
  defer(cudaFree(d_a));

  if (PRINT_IF_ERROR(cudaMalloc((void **) &d_b, b.size() * sizeof(*b.data())))) {
    LOG(critical, "CUDA/VECTOR_ADD/BASIC device memory allocation failed for vector B");
    state.SkipWithError("CUDA/VECTOR_ADD/BASIC device memory allocation failed for vector B");
    return;
  }
  defer(cudaFree(d_b));

  if (PRINT_IF_ERROR(cudaMalloc((void **) &d_c, c.size() * sizeof(*c.data())))) {
    LOG(critical, "CUDA/VECTOR_ADD/BASIC device memory allocation failed for vector C");
    state.SkipWithError("CUDA/VECTOR_ADD/BASIC device memory allocation failed for vector C");
    return;
  }
  defer(cudaFree(d_c));

  if (PRINT_IF_ERROR(cudaMemcpy(d_a, a.data(), a.size() * sizeof(*a.data()), cudaMemcpyHostToDevice))) {
    state.SkipWithError("CUDA/VECTOR_ADD/BASIC device memory copy failed for vector A");
    return;
  }

  if (PRINT_IF_ERROR(cudaMemcpy(d_b, b.data(), b.size() * sizeof(*b.data()), cudaMemcpyHostToDevice))) {
    state.SkipWithError("CUDA/VECTOR_ADD/BASIC device memory copy failed for vector B");
    return;
  }

  if (PRINT_IF_ERROR(cudaMemcpy(d_c, c.data(), c.size() * sizeof(*c.data()), cudaMemcpyHostToDevice))) {
    state.SkipWithError("CUDA/VECTOR_ADD/BASIC device memory copy failed for vector C");
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

    cuda_vector_add<T, COARSINING_FACTOR, BLOCK_SIZE><<<gridDim, blockDim>>>(d_a, d_b, d_c, N);

#ifdef USE_CUDA_EVENTS
    cudaEventRecord(stop, NULL);
    const auto cuda_err = cudaEventSynchronize(stop);
#else  // USE_CUDA_EVENTS
    const auto cuda_err = cudaDeviceSynchronize();
#endif // USE_CUDA_EVENTS

    state.PauseTiming();
    if (PRINT_IF_ERROR(cuda_err)) {
      state.SkipWithError("CUDA/VECTOR_ADD/BASIC failed to launch kernel");
      break;
    }
#ifdef USE_CUDA_EVENTS
    float msecTotal = 0.0f;
    if (PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop))) {
      state.SkipWithError("CUDA/VECTOR_ADD/BASIC failed to get elapsed time");
      break;
    }
    state.SetIterationTime(msecTotal / 1000);
#endif // USE_CUDA_EVENTS
    state.ResumeTiming();
  }

  state.counters.insert({{"N", N},
                         {"BLOCK_SIZE", BLOCK_SIZE},
                         {"Flops", {1.0 * N, benchmark::Counter::kAvgThreadsRate}},
                         {"COARSINING_FACTOR", COARSINING_FACTOR}});
  state.SetBytesProcessed(int64_t(state.iterations()) * N * sizeof(T));
  state.SetItemsProcessed(int64_t(state.iterations()) * N);
}

#ifdef USE_CUDA_EVENTS
#ifndef FAST_MODE
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, char, 1, 32)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, int, 1, 32)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, float, 1, 32)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, double, 1, 32)->ALL_ARGS()->UseManualTime();

BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, char, 1, 64)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, int, 1, 64)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, float, 1, 64)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, double, 1, 64)->ALL_ARGS()->UseManualTime();

BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, char, 1, 128)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, int, 1, 128)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, float, 1, 128)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, double, 1, 128)->ALL_ARGS()->UseManualTime();

BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, char, 1, 256)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, int, 1, 256)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, float, 1, 256)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, double, 1, 256)->ALL_ARGS()->UseManualTime();
#endif // FAST_MODE

BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, char, 1, 512)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, int, 1, 512)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, float, 1, 512)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, double, 1, 512)->ALL_ARGS()->UseManualTime();

#else // USE_CUDA_EVENTS
#ifndef FAST_MODE
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, char, 1, 32)->ALL_ARGS();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, int, 1, 32)->ALL_ARGS();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, float, 1, 32)->ALL_ARGS();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, double, 1, 32)->ALL_ARGS();

BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, char, 1, 64)->ALL_ARGS();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, int, 1, 64)->ALL_ARGS();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, float, 1, 64)->ALL_ARGS();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, double, 1, 64)->ALL_ARGS();

BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, char, 1, 128)->ALL_ARGS();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, int, 1, 128)->ALL_ARGS();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, float, 1, 128)->ALL_ARGS();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, double, 1, 128)->ALL_ARGS();

BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, char, 1, 256)->ALL_ARGS();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, int, 1, 256)->ALL_ARGS();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, float, 1, 256)->ALL_ARGS();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, double, 1, 256)->ALL_ARGS();

#endif // FAST_MODE

BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, char, 1, 512)->ALL_ARGS();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, int, 1, 512)->ALL_ARGS();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, float, 1, 512)->ALL_ARGS();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, double, 1, 512)->ALL_ARGS();

#endif // USE_CUDA_EVENTS
