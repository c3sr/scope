#include <benchmark/benchmark.h>

#include "ops.hpp"


void rd_8(char *ptr, const size_t count, const size_t stride)
{
    int64_t *dp = reinterpret_cast<int64_t*>(ptr);

    const size_t numElems = count / sizeof(int64_t);
    const size_t elemsPerStride = stride / sizeof(int64_t);

    int64_t acc = 0;
    #pragma omp parallel for schedule(static) private(acc)
    for (size_t i = 0; i < numElems; i += elemsPerStride)
    {
        benchmark::DoNotOptimize(acc += dp[i]);
    benchmark::ClobberMemory();
    }
}

void wr_8(char *ptr, const size_t count, const size_t stride)
{
    int64_t *dp = reinterpret_cast<int64_t*>(ptr);

    const size_t numElems = count / sizeof(int64_t);
    const size_t elemsPerStride = stride / sizeof(int64_t);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numElems; i += elemsPerStride)
    {
        benchmark::DoNotOptimize(dp[i] = i);
    benchmark::ClobberMemory();
    }
}
