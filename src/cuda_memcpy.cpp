#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include "init.hpp"
#include "utils.hpp"
#include "utils_cuda.hpp"

static void CUDAMemcpyToGPU(benchmark::State &state) {

  if (!has_cuda) {
    state.SkipWithError("CUDA/MEMCPY/TOGPU no CUDA device found");
    return;
  }

  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));
  char *src        = new char[bytes];
  char *dst        = nullptr;

  defer(delete[] src);
  const auto err = cudaMalloc(&dst, bytes);
  if (err != cudaSuccess) {
    state.SkipWithError("failed to perform cudaMemcpy");
    return;
  }
  defer(cudaFree(dst));

#ifdef USE_CUDA_EVENTS
  cudaEvent_t start, stop;
  CUDA_PERROR(cudaEventCreate(&start));
  CUDA_PERROR(cudaEventCreate(&stop));
#endif // USE_CUDA_EVENTS

  for (auto _ : state) {
#ifdef USE_CUDA_EVENTS
    cudaEventRecord(start, NULL);
#endif // USE_CUDA_EVENTS

    auto cuda_err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);

#ifdef USE_CUDA_EVENTS
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
#endif // USE_CUDA_EVENTS

    state.PauseTiming();

    if (CUDA_PERROR(cuda_err) != cudaSuccess) {
      state.SkipWithError("CUDA/MEMCPY/TOGPU failed to perform memcpy");
      break;
    }
#ifdef USE_CUDA_EVENTS
    float msecTotal = 0.0f;
    if ((cuda_err = CUDA_PERROR(cudaEventElapsedTime(&msecTotal, start, stop)))) {
      state.SkipWithError("CUDA/MEMCPY/TOGPU failed to get elapsed time");
    }
    state.SetIterationTime(msecTotal / 1000);
#endif // USE_CUDA_EVENTS
    state.ResumeTiming();
  }
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters.insert({{"bytes", bytes}});
}

static void CUDAPinnedMemcpyToGPU(benchmark::State &state) {

  if (!has_cuda) {
    state.SkipWithError("CUDA/PINNED_MEMCPY/TOGPU no CUDA device found");
    return;
  }

  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));
  float *src       = nullptr;
  auto err         = cudaHostAlloc(&src, bytes, cudaHostAllocWriteCombined);
  if (err != cudaSuccess) {
    state.SkipWithError("failed to perform pinned cudaHostAlloc");
    return;
  }
  defer(cudaFree(src));

  memset(src, 0, bytes);
  float *dst = nullptr;

  err = cudaMalloc(&dst, bytes);
  if (err != cudaSuccess) {
    state.SkipWithError("failed to perform cudaMalloc");
    return;
  }
  defer(cudaFree(dst));

#ifdef USE_CUDA_EVENTS
  cudaEvent_t start, stop;
  CUDA_PERROR(cudaEventCreate(&start));
  CUDA_PERROR(cudaEventCreate(&stop));
#endif // USE_CUDA_EVENTS

  for (auto _ : state) {
#ifdef USE_CUDA_EVENTS
    cudaEventRecord(start, NULL);
#endif // USE_CUDA_EVENTS

    auto cuda_err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);

#ifdef USE_CUDA_EVENTS
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
#endif // USE_CUDA_EVENTS

    state.PauseTiming();

    if (CUDA_PERROR(cuda_err) != cudaSuccess) {
      state.SkipWithError("CUDA/PINNED_MEMCPY/TOGPU failed to perform memcpy");
      break;
    }
#ifdef USE_CUDA_EVENTS
    float msecTotal = 0.0f;
    if ((cuda_err = CUDA_PERROR(cudaEventElapsedTime(&msecTotal, start, stop)))) {
      state.SkipWithError("CUDA/PINNED_MEMCPY/TOGPU failed to get elapsed time");
      break;
    }
    state.SetIterationTime(msecTotal / 1000);
#endif // USE_CUDA_EVENTS
    state.ResumeTiming();
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters.insert({{"bytes", bytes}});
}

#ifdef USE_CUDA_EVENTS
BENCHMARK(CUDAMemcpyToGPU)->DenseRange(1, 31, 1)->UseManualTime();
BENCHMARK(CUDAPinnedMemcpyToGPU)->DenseRange(1, 31, 1)->UseManualTime();
#else  // USE_CUDA_EVENTS
BENCHMARK(CUDAMemcpyToGPU)->DenseRange(1, 31, 1);
BENCHMARK(CUDAPinnedMemcpyToGPU)->DenseRange(1, 31, 1);
#endif // USE_CUDA_EVENTS
