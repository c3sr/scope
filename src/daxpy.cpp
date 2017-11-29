#include <benchmark/benchmark.h>

#include <cblas.h>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

static void DAXPY(benchmark::State &state) {
  const auto N = state.range(0);
  const auto x_incr = state.range(1);
  const auto y_incr = state.range(2);
  const auto alpha = 0.5;

    auto x = std::vector<double>(N);
    auto y = std::vector<double>(N);
    std::iota(x.begin(), x.end(), 1);
    std::iota(y.begin(), y.end(), 1);
  for (auto _ : state) {
    cblas_daxpy(N, alpha, x.data(), x_incr, y.data(), y_incr); 
  }
 
  state.counters.insert({{"N", N}, {"x_increment", x_incr}, {"y_increment", y_incr}});
  state.SetBytesProcessed(int64_t(state.iterations()) * 3*8*N);
}

BENCHMARK(DAXPY) // N, DA, INCX, INCY
    ->Args({10, 1, 1})
    ->Args({100, 1, 1})
    ->Args({1000, 1, 1})
    ->Args({10000, 1, 1})
    ->Args({100000, 1, 1})
    ->Args({10000000, 1, 1})
    ->Args({100000000, 1, 1})
    ->Args({1000000000, 1, 1})
    ->Unit(benchmark::kNanosecond);
