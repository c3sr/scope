#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>
#include <numa.h>

#include "init/init.hpp"
#include "utils/utils.hpp"

#include "cuda-numa/args.hpp"

#define NAME "NUMAUM/Coherence/GpuToHost"

static void cpu_access(char *ptr, const size_t n, const size_t stride) {
  for (size_t i = 0; i < n; i += stride) {
    benchmark::DoNotOptimize(ptr[i] = 0);
  }
}

static void NUMAUM_Direct_GPUToHost(benchmark::State &state) {

  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  if (!has_numa) {
    state.SkipWithError(NAME " NUMA not available");
    return;
  }

  const int numa_id = state.range(1);
  const int cuda_id = state.range(2);

  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));


  if (0 != numa_run_on_node(numa_id)) {
    state.SkipWithError(NAME " couldn't bind to NUMA node");
    return;
  }

  if (PRINT_IF_ERROR(cudaSetDevice(cuda_id))) {
    state.SkipWithError(NAME " failed to set CUDA device");
    return;
  }
  if (PRINT_IF_ERROR(cudaDeviceReset())) {
    state.SkipWithError(NAME " failed to reset device");
    return;
  }

  char *ptr = nullptr;
  if (PRINT_IF_ERROR(cudaMallocManaged(&ptr, bytes))) {
    state.SkipWithError(NAME " failed to perform cudaMallocManaged");
    return;
  }
  defer(cudaFree(ptr));

  if (PRINT_IF_ERROR(cudaMemset(ptr, 0, bytes))) {
    state.SkipWithError(NAME " failed to perform cudaMemset");
    return;
  }

  for (auto _ : state) {
    // state.PauseTiming();
    if (PRINT_IF_ERROR(cudaMemPrefetchAsync(ptr, bytes, cuda_id))) {
      state.SkipWithError(NAME " failed to perform cudaMemPrefetch");
      return;
    }
    if (PRINT_IF_ERROR(cudaDeviceSynchronize())) {
      state.SkipWithError(NAME " failed to synchronize");
      return;
    }
    // state.ResumeTiming();



    auto start = std::chrono::high_resolution_clock::now();
    cpu_access(ptr, bytes, 4096);
    auto end   = std::chrono::high_resolution_clock::now();

    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters.insert({{"bytes", bytes}});

  // reset to run on any node
  if (0 != numa_run_on_node(-1)) {
    LOG(critical, NAME " couldn't allow bindings to all nodes");
    exit(-1);
  }
}

BENCHMARK(NUMAUM_Direct_GPUToHost)->Apply(CustomArguments)->MinTime(0.1)->UseManualTime();
