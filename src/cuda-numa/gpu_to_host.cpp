#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>
#include <numa.h>

#include "init/init.hpp"
#include "utils/utils.hpp"

#include "cuda-numa/args.hpp"

static void CUDANUMA_Memcpy_GPUToHost(benchmark::State &state) {

  if (!has_cuda) {
    state.SkipWithError("CUDANUMA/MEMCPY/GPUToHost no CUDA device found");
    return;
  }

  if (!has_numa) {
    state.SkipWithError("CUDANUMA/MEMCPY/GPUToHost NUMA not available");
    return;
  }

  const int numa_id = state.range(1);
  const int cuda_id = state.range(2);

  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));
  char *src        = nullptr;
  char *dst        = new char[bytes];

  defer(delete[] dst);

  if (0 != numa_run_on_node(numa_id)) {
    state.SkipWithError("CUDANUMA/MEMCPY/GPUToHost couldn't bind to NUMA node");
    return;
  }


  if (PRINT_IF_ERROR(cudaSetDevice(cuda_id))) {
    state.SkipWithError("CUDANUMA/MEMCPY/GPUToHost failed to set CUDA device");
    return;
  }

  if (PRINT_IF_ERROR(cudaMalloc(&src, bytes))) {
    state.SkipWithError("CUDANUMA/MEMCPY/GPUToHost failed to perform cudaMalloc");
    return;
  }
  defer(cudaFree(src));

  if (PRINT_IF_ERROR(cudaMemset(src, 0, bytes))) {
    state.SkipWithError("CUDANUMA/MEMCPY/GPUToHost failed to perform cudaMemset");
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

    const auto cuda_err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);

#ifdef USE_CUDA_EVENTS
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
#endif // USE_CUDA_EVENTS

    state.PauseTiming();

    if (PRINT_IF_ERROR(cuda_err) != cudaSuccess) {
      state.SkipWithError("CUDANUMA/MEMCPY/GPUToHost failed to perform memcpy");
      break;
    }
#ifdef USE_CUDA_EVENTS
    float msecTotal = 0.0f;
    if (PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop))) {
      state.SkipWithError("CUDANUMA/MEMCPY/GPUToHost failed to get elapsed time");
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
BENCHMARK(CUDANUMA_Memcpy_GPUToHost)->TEST_ARGS()->UseManualTime();
// BENCHMARK(CUDANUMA_Memcpy_GPUToHost)->ALL_ARGS()->UseManualTime();
#else  // USE_CUDA_EVENTS
BENCHMARK(CUDANUMA_Memcpy_GPUToHost)->ALL_ARGS();
#endif // USE_CUDA_EVENTS
