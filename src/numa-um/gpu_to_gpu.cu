#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>
#include <numa.h>

#include "init/init.hpp"
#include "utils/utils.hpp"

#include "numa-um/args.hpp"

#define NAME "NUMAUM/Coherence/GPUToGPU"

template <bool NOOP = false>
__global__ void gpu_write(char *ptr, const size_t count, const size_t stride)
{
  if (NOOP)
  {
    return;
  }

  // global ID
  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  // lane ID 0-31
  const size_t lx = gx & 31;
  // warp ID
  size_t wx = gx / 32;
  const size_t numWarps = (gridDim.x * blockDim.x + 32 - 1) / 32;

  if (0 == lx)
  {
    for (size_t i = wx * stride; i < count; i += numWarps * stride)
    {
      ptr[i] = 0;
    }
  }
}

static void NUMAUM_Direct_GPUToGPU(benchmark::State &state) {

  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  if (!has_numa) {
    state.SkipWithError(NAME " NUMA not available");
    return;
  }

  const size_t pageSize = page_size();

  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));
  const int src_gpu = state.range(1);
  const int dst_gpu = state.range(2);

  if (PRINT_IF_ERROR(cudaSetDevice(src_gpu))) {
    state.SkipWithError(NAME " failed to set CUDA src device");
    return;
  }
  if (PRINT_IF_ERROR(cudaDeviceReset())) {
    state.SkipWithError(NAME " failed to reset device");
    return;
  }
  if (PRINT_IF_ERROR(cudaSetDevice(dst_gpu))) {
    state.SkipWithError(NAME " failed to set CUDA dst device");
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

  cudaEvent_t start, stop;
  if (PRINT_IF_ERROR(cudaEventCreate(&start))) {
    state.SkipWithError(NAME " failed to create event");
    return;
  }
  defer(cudaEventDestroy(start));

  cudaEventCreate(&stop);
  defer(cudaEventDestroy(stop));

  for (auto _ : state) {
    // state.PauseTiming();
    cudaMemPrefetchAsync(ptr, bytes, src_gpu);
    cudaSetDevice(src_gpu);
    cudaDeviceSynchronize();
    cudaSetDevice(dst_gpu);
    cudaDeviceSynchronize();
    if (PRINT_IF_ERROR(cudaGetLastError())) {
      state.SkipWithError(NAME " failed to prep iteration");
      return;
    }
    // state.ResumeTiming();


    cudaEventRecord(start);
    gpu_write<<<256,256>>>(ptr, bytes, pageSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float millis = 0;
    if (PRINT_IF_ERROR(cudaEventElapsedTime(&millis, start, stop))) {
      state.SkipWithError("CUDA/MEMCPY/HostToGPU failed to get elapsed time");
      break;
    }
    state.SetIterationTime(millis / 1000);
    // cudaDeviceSynchronize();

  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters.insert({{"bytes", bytes}});

}

BENCHMARK(NUMAUM_Direct_GPUToGPU)->Apply(ArgsCountGpuGpuNoSelf)->MinTime(0.1)->UseRealTime()->UseManualTime();
