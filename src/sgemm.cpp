#include <benchmark/benchmark.h>

#include <cblas.h>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

static void SGEMM(benchmark::State &state) {
  const auto M = state.range(0);
  const auto N = state.range(1);
  const auto K = state.range(2);
  const auto alpha = state.range(3);
  const auto beta = state.range(4);

  for (auto _ : state) {
    state.PauseTiming();
    auto a = std::vector<float>(M * K);
    auto b = std::vector<float>(K * N);
    auto c = std::vector<float>(M * N);
    std::iota(a.begin(), a.end(), 1);
    std::iota(b.begin(), b.end(), 1);
    std::fill(c.begin(), c.end(), 0);
    state.ResumeTiming();

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha,
                a.data(), K, b.data(), N, beta, c.data(), N);
  }

  state.counters.insert(
      {{"M", M}, {"N", N}, {"K", K}, {"alpha", alpha}, {"beta", beta}});
}

BENCHMARK(SGEMM) // M, N, K , alpha, beta
    ->Args({1000, 1, 1, -1, 1})
    ->Args({128, 169, 1728, 1, 0})
    ->Args({128, 729, 1200, 1, 0})
    ->Args({192, 169, 1728, 1, 0})
    ->Args({256, 169, 1, 1, 1})
    ->Args({256, 729, 1, 1, 1})
    ->Args({384, 169, 1, 1, 1})
    ->Args({384, 169, 2304, 1, 0})
    ->Args({50, 1000, 1, 1, 1})
    ->Args({50, 1000, 4096, 1, 0})
    ->Args({50, 4096, 1, 1, 1})
    ->Args({50, 4096, 4096, 1, 0})
    ->Args({50, 4096, 9216, 1, 0})
    ->Args({96, 3025, 1, 1, 1})
    ->Args({96, 3025, 363, 1, 0});
