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


    const auto bytes = (1ULL << static_cast<size_t>(state.range(0))) / state.threads;
    const int src_numa = state.range(1);
    const int dst_numa = state.range(2);

  // Setup
    const long pageSize = sysconf(_SC_PAGESIZE);
    numa_bind_node(src_numa);
    char *ptr = static_cast<char *>(aligned_alloc(pageSize, bytes));

    std::memset(ptr, 0, bytes);
    benchmark::DoNotOptimize(ptr);

  for (auto _ : state) {
    state.PauseTiming();
    // invalidate data in dst cache
    numa_bind_node(src_numa);
    //for (size_t i = 0; i < bytes; ++i) {
    //   ptr[i] = rand();
    //}
    benchmark::ClobberMemory();
    std::memset(ptr, 0, bytes);
    benchmark::ClobberMemory();

    // Access from Device and Time
    numa_bind_node(dst_numa);
    state.ResumeTiming();

    rd_8(ptr, bytes, 8);

  }

  numa_bind_node(-1);

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
    state.counters.insert({{"bytes", bytes}});

    free(ptr);

}


BENCHMARK(NUMA_RD)->ThreadRange(1,8)->Apply(ArgsCountNumaNuma)->MinTime(0.1)->UseRealTime();
