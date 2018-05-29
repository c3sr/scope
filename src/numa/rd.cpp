#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <omp.h>
#include <numa.h>

#include "init/init.hpp"
#include "utils/utils.hpp"

#include "numa/args.hpp"

#include "ops.hpp"

#define NAME "NUMA/RD"

static void NUMA_RD(benchmark::State &state) {

    if (!has_numa) {
        state.SkipWithError(NAME " NUMA not available");
        return;
    }


    const size_t threads = state.range(0);
    const auto bytes = 1ULL << static_cast<size_t>(state.range(1));
    const int src_numa = state.range(2);
    const int dst_numa = state.range(3);

  // Setup
    const long pageSize = sysconf(_SC_PAGESIZE);
    omp_numa_bind_node(src_numa);
    char *ptr = static_cast<char *>(aligned_alloc(pageSize, bytes));

    std::memset(ptr, 0, bytes);
    benchmark::DoNotOptimize(ptr);

  for (auto _ : state) {
    state.PauseTiming();
    // invalidate data in dst cache
    omp_numa_bind_node(src_numa);
    //for (size_t i = 0; i < bytes; ++i) {
    //   ptr[i] = rand();
    //}
    benchmark::ClobberMemory();
    std::memset(ptr, 0, bytes);
    benchmark::ClobberMemory();

    // Access from Device and Time
    omp_numa_bind_node(dst_numa);
    state.ResumeTiming();

    rd_8(ptr, bytes, 8);

  }

  omp_numa_bind_node(-1);

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
    state.counters.insert({{"bytes", bytes}});

    free(ptr);

}


BENCHMARK(NUMA_RD)->Apply(ArgsThreadCountNumaNuma)->MinTime(0.1)->UseRealTime();
