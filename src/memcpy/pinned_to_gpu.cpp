
#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>

#include "init/init.hpp"
#include "utils/utils.hpp"

#include "memcpy/args.hpp"

static void CUDA_Memcpy_PinnedToGPU(benchmark::State &state) {

  const auto flags = cudaHostAllocWriteCombined;

  state.SetLabel(fmt::format("CUDA/Memcpy/PinnedToGPU/{}", "cudaHostAllocWriteCombined"));

  if (!has_cuda) {
    state.SkipWithError("CUDA/MEMCPY/PinnedToGPU no CUDA device found");
    return;
  }

  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));
  char *src        = nullptr;
  if (PRINT_IF_ERROR(cudaHostAlloc(&src, bytes, flags))) {
    state.SkipWithError("CUDA/MEMCPY/PinnedToGPU failed to perform pinned cudaHostAlloc");
    return;
  }
  defer(cudaFreeHost(src));

  memset(src, 0, bytes);

  char *dst = nullptr;
  if (PRINT_IF_ERROR(cudaMalloc(&dst, bytes))) {
    state.SkipWithError("CUDA/MEMCPY/PinnedToGPU failed to perform cudaMalloc");
    return;
  }
  defer(cudaFree(dst));

#ifdef USE_CUDA_EVENTS
  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));
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

    if (PRINT_IF_ERROR(cuda_err)) {
      state.SkipWithError("CUDA/MEMCPY/PinnedToGPU failed to perform memcpy");
      break;
    }
#ifdef USE_CUDA_EVENTS
    float msecTotal = 0.0f;
    if (PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop))) {
      state.SkipWithError("CUDA/MEMCPY/PinnedToGPU failed to get elapsed time");
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
BENCHMARK(CUDA_Memcpy_PinnedToGPU)->ALL_ARGS()->UseManualTime();
#else  // USE_CUDA_EVENTS
BENCHMARK(CUDA_Memcpy_PinnedToGPU)->ALL_ARGS();
#endif // USE_CUDA_EVENTS
