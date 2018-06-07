#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>

#include "init/init.hpp"
#include "utils/utils.hpp"

#include "memcpy-duplex/args.hpp"

#define NAME "DUPLEX/Memcpy/GPUGPU"

static void DUPLEX_Memcpy_GPUGPU(benchmark::State &state) {

  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));
  const int gpu0 = state.range(1);
  const int gpu1 = state.range(2);

  if (PRINT_IF_ERROR(utils::cuda_reset_device(gpu0))) {
    state.SkipWithError(NAME " failed to reset CUDA device");
    return;
  }
  if (PRINT_IF_ERROR(utils::cuda_reset_device(gpu1))) {
    state.SkipWithError(NAME " failed to reset CUDA device");
    return;
  }

  // There are two copies, one gpu0 -> gpu1, one gpu1 -> gpu0

  // Create One stream per copy
  std::vector<cudaStream_t> streams;
  // cudaStreamCreate(...)

  // Start and stop events for each copy
  std::vector<cudaEvent_t> starts;
  std::vector<cudaEvent_t> stops;
  // cudaEventCreate(...)

  // Source and destination for each copy
  std::vector<char *> srcs;
  std::vector<char *> dsts;

  // create a source and destination allocation on each gpu
  for (auto gpu : {gpu0, gpu1} ) {
  // Set to the 
  if (PRINT_IF_ERROR(cudaSetDevice(gpu))) {
    state.SkipWithError(NAME " failed to set src device");
    return;
  }
  // create a src allocation on gpup0
  char *ptr;
  if (PRINT_IF_ERROR(cudaMalloc(&ptr, bytes))) {
    state.SkipWithError(NAME " failed to perform cudaMalloc");
    return;
  }
  defer(cudaFree(ptr));
  srcs.push_back(ptr);
  if (PRINT_IF_ERROR(cudaMemset(ptr, 0, bytes))) {
    state.SkipWithError(NAME " failed to perform src cudaMemset");
    return;
  }
  // create a destination allocation on gpu
  // ...
  }


  assert(starts.size() == stops.size());
  assert(streams.size() == starts.size());
  assert(srcs.size() == dsts.size());
  assert(streams.size() == srcs.size());

  for (auto _ : state) {

    // Start all copies
    for (size_t i = 0; i < streams.size(); ++i) {
      auto start = starts[i];
      auto stop = stops[i];
      auto stream = streams[i];
      auto src = srcs[i];
      auto dst = dsts[i];
      cudaEventRecord(start, stream);
      cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
      cudaEventRecord(stop, stream);
    }

    // Wait for all copies to finish
    for (auto s : stops) {
      if (PRINT_IF_ERROR(cudaEventSynchronize(s))) {
        state.SkipWithError(NAME " failed to synchronize");
        return;
      }
    }

    // Find the longest time between any start and stop
    float maxMillis = 0;
    for (const auto start : starts) {
      for (const auto stop : stops) {
        float millis;
        cudaEventElapsedTime(&millis, start, stop);
        maxMillis = std::max(millis, maxMillis);
      }
    }


    state.SetIterationTime(maxMillis / 1000);
  }
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters.insert({{"bytes", bytes}});
}

BENCHMARK(DUPLEX_Memcpy_GPUGPU)->Apply(ArgsCountGpuGpuNoSelf)->UseManualTime();
