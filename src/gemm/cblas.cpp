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
static void cblas_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N,
                       const int K, const T alpha, const T* A, const T* B, const T beta, T* C);

template <>
void cblas_gemm<float>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N,
                       const int K, const float alpha, const float* A, const float* B, const float beta, float* C) {

  const int lda = (TransA == CblasNoTrans) ? K : M;
  const int ldb = (TransB == CblasNoTrans) ? N : K;

  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
}

template <>
void cblas_gemm<double>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N,
                        const int K, const double alpha, const double* A, const double* B, const double beta,
                        double* C) {

  const int lda = (TransA == CblasNoTrans) ? K : M;
  const int ldb = (TransB == CblasNoTrans) ? N : K;

  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
}

template <>
void cblas_gemm<std::complex<float>>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M,
                                     const int N, const int K, const std::complex<float> alpha,
                                     const std::complex<float>* A, const std::complex<float>* B,
                                     const std::complex<float> beta, std::complex<float>* C) {

  using scalar_type = float;

  const int lda = (TransA == CblasNoTrans) ? K : M;
  const int ldb = (TransB == CblasNoTrans) ? N : K;

  cblas_cgemm(CblasRowMajor, TransA, TransB, M, N, K, reinterpret_cast<const scalar_type(&)[2]>(alpha),
              reinterpret_cast<const scalar_type*>(A), lda, reinterpret_cast<const scalar_type*>(B), ldb,
              reinterpret_cast<const scalar_type(&)[2]>(beta), reinterpret_cast<scalar_type*>(C), N);
}

template <>
void cblas_gemm<std::complex<double>>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M,
                                      const int N, const int K, const std::complex<double> alpha,
                                      const std::complex<double>* A, const std::complex<double>* B,
                                      const std::complex<double> beta, std::complex<double>* C) {

  using scalar_type = double;

  const int lda = (TransA == CblasNoTrans) ? K : M;
  const int ldb = (TransB == CblasNoTrans) ? N : K;

  cblas_zgemm(CblasRowMajor, TransA, TransB, M, N, K, reinterpret_cast<const scalar_type(&)[2]>(alpha),
              reinterpret_cast<const scalar_type*>(A), lda, reinterpret_cast<const scalar_type*>(B), ldb,
              reinterpret_cast<const scalar_type(&)[2]>(beta), reinterpret_cast<scalar_type*>(C), N);
}

template <typename T>
static void CBLAS(benchmark::State& state) {

  static const std::string IMPLEMENTATION_NAME = gemm::detail::implementation_name<T>();
  state.SetLabel(fmt::format("CBLAS/{}", IMPLEMENTATION_NAME));

  const T one  = gemm::detail::one<T>();
  const T zero = gemm::detail::zero<T>();

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
    cblas_gemm<T>(CblasNoTrans, CblasNoTrans, M, N, K, alpha, a.data(), b.data(), beta, c.data());
    benchmark::DoNotOptimize(c.data());
  }

  state.counters.insert(
      {{"M", M}, {"N", N}, {"K", K}, {"Flops", {2.0 * M * N * K, benchmark::Counter::kAvgThreadsRate}}});
  state.SetBytesProcessed(int64_t(state.iterations()) * a.size() * b.size() * c.size());
  state.SetItemsProcessed(int64_t(state.iterations()) * M * N * K);
}

static void CBLAS_SGEMM(benchmark::State& state) {
  return CBLAS<float>(state);
}

static void CBLAS_DGEMM(benchmark::State& state) {
  return CBLAS<double>(state);
}

static void CBLAS_CGEMM(benchmark::State& state) {
  return CBLAS<std::complex<float>>(state);
}

static void CBLAS_ZGEMM(benchmark::State& state) {
  return CBLAS<std::complex<double>>(state);
}

// BENCHMARK(CBLAS_SGEMM)->ALL_ARGS();
// BENCHMARK(CBLAS_DGEMM)->ALL_ARGS();
// BENCHMARK(CBLAS_CGEMM)->ALL_ARGS();
// BENCHMARK(CBLAS_ZGEMM)->ALL_ARGS();
