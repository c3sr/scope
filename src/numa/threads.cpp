#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <omp.h>
#include <numa.h>

#include "init/init.hpp"
#include "utils/utils.hpp"

#include "numa/args.hpp"

#define NAME "THREADS"

static void cpu_read_8(char * ptr, const size_t count, const size_t stride)
{
    assert(ptr);
    int64_t *dp = reinterpret_cast<int64_t*>(ptr);

    const size_t numElems = count / sizeof(int64_t);
    const size_t elemsPerStride = stride / sizeof(int64_t);

    int64_t acc = 0;
    for (size_t i = 0; i < numElems; i += elemsPerStride)
    {
        benchmark::DoNotOptimize(acc += dp[i]);
    }
    benchmark::ClobberMemory();
}

static void THREADS(benchmark::State &state) {

    if (!has_numa) {
        state.SkipWithError(NAME " NUMA not available");
        return;
    }


    const auto bytes = (1ULL << static_cast<size_t>(state.range(0))) / state.threads;
    const int src_numa = state.range(1);
    const int dst_numa = state.range(2);

  // Setup
    const long pageSize = sysconf(_SC_PAGESIZE);
    numa_bind_node(src_numa);
    char *ptr = static_cast<char *>(aligned_alloc(pageSize, bytes));
    std::memset(ptr, 0, bytes);



  for (auto _ : state) {
    state.PauseTiming();
    // invalidate data in dst cache
    numa_bind_node(src_numa);
    std::memset(ptr, 0, bytes);

    // Access from Device and Time
    numa_bind_node(dst_numa);
    state.ResumeTiming();

    cpu_read_8(ptr, bytes, 8);

  }

  numa_bind_node(-1);

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
    state.counters.insert({{"bytes", bytes}});

    free(ptr);

}


BENCHMARK(THREADS)->ThreadRange(1,8)->Apply(ArgsCountNumaNuma)->MinTime(0.1)->UseRealTime();
