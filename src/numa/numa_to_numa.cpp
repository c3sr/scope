#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <omp.h>
#include <numa.h>

#include "init/init.hpp"
#include "utils/utils.hpp"

#include "numa/args.hpp"

#define NAME "NUMA/OMP/CpuToCpu"

static void cpu_read_8(double *__restrict__ ptr, const size_t count, const size_t stride)
{

    const size_t numElems = count / sizeof(double);
    const size_t elemsPerStride = stride / sizeof(double);

    double acc = 0;
#pragma omp parallel for schedule(static) private(acc)
    for (size_t i = 0; i < numElems; i += elemsPerStride)
    {
        benchmark::DoNotOptimize(acc += ptr[i]);
    }
}

static void NUMAOMP_RD_CpuToCpu(benchmark::State &state) {

  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  if (!has_numa) {
    state.SkipWithError(NAME " NUMA not available");
    return;
  }

  const long pageSize = sysconf(_SC_PAGESIZE);

  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));
  const size_t num_threads = state.range(1);
  const int src_numa = state.range(2);
  const int dst_numa = state.range(3);

  omp_set_num_threads(num_threads);
  if (omp_get_max_threads() != num_threads) {
      std::cerr <<  num_threads << " " <<  omp_get_num_threads();
      state.SkipWithError(NAME " failed to set OMP threads");
      return;
  }
  
  omp_numa_bind_node(src_numa);
  double *ptr = static_cast<double *>(aligned_alloc(pageSize, bytes));
  defer(free(ptr));
  std::memset(ptr, 0, bytes);

  omp_numa_bind_node(dst_numa);

  for (auto _ : state) {
    state.PauseTiming();
    // invalidate data in dst cache
    omp_numa_bind_node(src_numa);
    std::memset(ptr, 0, bytes);

    // Access from Device and Time
    omp_numa_bind_node(dst_numa);
    state.ResumeTiming();

    cpu_read_8(ptr, bytes, 8);

  }
  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters.insert({{"bytes", bytes}});

  // reset to run on any node
  omp_numa_bind_node(-1);
}


BENCHMARK(NUMAOMP_RD_CpuToCpu)->Apply(ArgsCountThreadsNumaNuma)->UseRealTime();
