#include <benchmark/benchmark.h>

#include <complex>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <cblas.h>

#include "utils/utils.hpp"

#include "gemm/args.hpp"
#include "gemm/utils.hpp"

template <typename T>
static void CBLAS(benchmark::State &state) {

  static const std::string IMPLEMENTATION_NAME = gemm::detail::implementation_name<T>();

  const T one{1};
  const T zero{0};

  const auto M = state.range(0);
  const auto N = state.range(1);
  const auto K = state.range(2);
  T alpha{one};
  T beta{zero};

  auto a = std::vector<T>(M * K);
  auto b = std::vector<T>(K * N);
  auto c = std::vector<T>(M * N);
  std::fill(a.begin(), a.end(), one);
  std::fill(b.begin(), b.end(), one);
  std::fill(c.begin(), c.end(), zero);

  for (auto _ : state) {
    if constexpr (std::is_same<T, float>::value) {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, a.data(), K, b.data(), N, beta, c.data(),
                  N);
    } else if constexpr (std::is_same<T, double>::value) {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, a.data(), K, b.data(), N, beta, c.data(),
                  N);
    } else if constexpr (std::is_same<T, std::complex<float>>::value) {
      using scalar_type = typename T::value_type;
      cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, reinterpret_cast<scalar_type(&)[2]>(alpha),
                  reinterpret_cast<scalar_type *>(a.data()), K, reinterpret_cast<scalar_type *>(b.data()), N,
                  reinterpret_cast<scalar_type(&)[2]>(beta), reinterpret_cast<scalar_type *>(c.data()), N);
    } else if constexpr (std::is_same<T, std::complex<double>>::value) {
      using scalar_type = typename T::value_type;
      cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, reinterpret_cast<scalar_type(&)[2]>(alpha),
                  reinterpret_cast<scalar_type *>(a.data()), K, reinterpret_cast<scalar_type *>(b.data()), N,
                  reinterpret_cast<scalar_type(&)[2]>(beta), reinterpret_cast<scalar_type *>(c.data()), N);
    }
  }

  state.counters.insert(
      {{"M", M}, {"N", N}, {"K", K}, {"Flops", {2.0 * M * N * K, benchmark::Counter::kAvgThreadsRate}}});
  state.SetLabel(IMPLEMENTATION_NAME);
  state.SetBytesProcessed(int64_t(state.iterations()) * a.size() * b.size() * c.size());
  state.SetItemsProcessed(int64_t(state.iterations()) * M * N * K);
}

static void CBLAS_SGEMM(benchmark::State &state) {
  return CBLAS<float>(state);
}

static void CBLAS_DGEMM(benchmark::State &state) {
  return CBLAS<double>(state);
}

static void CBLAS_CGEMM(benchmark::State &state) {
  return CBLAS<std::complex<float>>(state);
}

static void CBLAS_ZGEMM(benchmark::State &state) {
  return CBLAS<std::complex<double>>(state);
}

BENCHMARK(CBLAS_SGEMM)->ALL_ARGS();
BENCHMARK(CBLAS_DGEMM)->ALL_ARGS();
BENCHMARK(CBLAS_CGEMM)->ALL_ARGS();
BENCHMARK(CBLAS_ZGEMM)->ALL_ARGS();
