#include <benchmark/benchmark.h>

#include <cblas.h>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "utils_sgemm.hpp"

static void CBLAS_SGEMM(benchmark::State &state) {
  const auto M = state.range(0);
  const auto N = state.range(1);
  const auto K = state.range(2);
  const auto alpha = 1.0f;
  const auto beta = 0.0f;

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

  state.counters.insert({{"M", M}, {"N", N}, {"K", K}});
  state.SetBytesProcessed(int64_t(state.iterations()) * 2 * M * N * K);
}

BENCHMARK(CBLAS_SGEMM)->SGEMM_ARGS();
