#include <benchmark/benchmark.h>

#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <cblas.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "init/init.hpp"
#include "utils/utils.hpp"

#include "gemm/args.hpp"
#include "gemm/utils.hpp"

template <typename T, typename Function>
static cublasStatus_t cublas_gemm_proxy(cublasHandle_t cublas_handle, Function F, const CBLAS_TRANSPOSE TransA,
                                        const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                                        const T* alpha, const T* A, const T* B, const T* beta, T* C) {

  const int lda              = (TransA == CblasNoTrans) ? K : M;
  const int ldb              = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB = (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  // Use the fact that C^T = (B^T . A^T)^T for optimization
  return F(cublas_handle, cuTransB, cuTransA, N, M, K, alpha, B, ldb, A, lda, beta, C, N);
}

template <typename T>
static cublasStatus_t cublas_gemm(cublasHandle_t cublas_handle, const CBLAS_TRANSPOSE TransA,
                                  const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const T* alpha,
                                  const T* A, const T* B, const T* beta, T* C);

template <>
cublasStatus_t cublas_gemm<__half>(cublasHandle_t cublas_handle, const CBLAS_TRANSPOSE TransA,
                                   const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                                   const __half* alpha, const __half* A, const __half* B, const __half* beta,
                                   __half* C) {
  return cublas_gemm_proxy<__half>(cublas_handle, cublasHgemm, TransA, TransB, M, N, K, alpha, A, B, beta, C);
}

template <>
cublasStatus_t cublas_gemm<float>(cublasHandle_t cublas_handle, const CBLAS_TRANSPOSE TransA,
                                  const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                                  const float* alpha, const float* A, const float* B, const float* beta, float* C) {

  return cublas_gemm_proxy<float>(cublas_handle, cublasSgemm, TransA, TransB, M, N, K, alpha, A, B, beta, C);
}

template <>
cublasStatus_t cublas_gemm<double>(cublasHandle_t cublas_handle, const CBLAS_TRANSPOSE TransA,
                                   const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                                   const double* alpha, const double* A, const double* B, const double* beta,
                                   double* C) {

  return cublas_gemm_proxy<double>(cublas_handle, cublasDgemm, TransA, TransB, M, N, K, alpha, A, B, beta, C);
}

template <>
cublasStatus_t cublas_gemm<cuComplex>(cublasHandle_t cublas_handle, const CBLAS_TRANSPOSE TransA,
                                      const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                                      const cuComplex* alpha, const cuComplex* A, const cuComplex* B,
                                      const cuComplex* beta, cuComplex* C) {

  return cublas_gemm_proxy<cuComplex>(cublas_handle, cublasCgemm, TransA, TransB, M, N, K, alpha, A, B, beta, C);
}

template <>
cublasStatus_t cublas_gemm<cuDoubleComplex>(cublasHandle_t cublas_handle, const CBLAS_TRANSPOSE TransA,
                                            const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                                            const cuDoubleComplex* alpha, const cuDoubleComplex* A,
                                            const cuDoubleComplex* B, const cuDoubleComplex* beta, cuDoubleComplex* C) {

  return cublas_gemm_proxy<cuDoubleComplex>(cublas_handle, cublasZgemm, TransA, TransB, M, N, K, alpha, A, B, beta, C);
}

template <typename T>
static void CUBLAS(benchmark::State& state) {
  static const std::string IMPLEMENTATION_NAME = gemm::detail::implementation_name<T>();
  state.SetLabel(fmt::format("CUBLAS/{}", IMPLEMENTATION_NAME));

  if (!has_cuda) {
    state.SkipWithError("CUDA/SGEMM no CUDA device found");
    return;
  }

  const T one  = gemm::detail::one<T>();
  const T zero = gemm::detail::zero<T>();

  const auto M = state.range(0);
  const auto N = state.range(1);
  const auto K = state.range(2);
  T alpha      = one;
  T beta       = zero;

  auto a = std::vector<T>(M * K);
  auto b = std::vector<T>(K * N);
  auto c = std::vector<T>(M * N);

  std::fill(a.begin(), a.end(), one);
  std::fill(b.begin(), b.end(), one);
  std::fill(c.begin(), c.end(), zero);

  using device_type = typename gemm::detail::cuda_type<T>::type;

  device_type *d_a{nullptr}, *d_b{nullptr}, *d_c{nullptr};

  if (PRINT_IF_ERROR(cudaMalloc((void**) &d_a, a.size() * sizeof(*a.data())))) {
    LOG(critical, "CUBLAS/{} device memory allocation failed for matrix A", IMPLEMENTATION_NAME);
    state.SkipWithError(
        fmt::format("CUBLAS/{} device memory allocation failed for matrix A", IMPLEMENTATION_NAME).c_str());
    return;
  }
  defer(cudaFree(d_a));

  if (PRINT_IF_ERROR(cudaMalloc((void**) &d_b, b.size() * sizeof(*b.data())))) {
    LOG(critical, "CUBLAS/{} device memory allocation failed for matrix B", IMPLEMENTATION_NAME);
    state.SkipWithError(
        fmt::format("CUBLAS/{} device memory allocation failed for matrix B", IMPLEMENTATION_NAME).c_str());
    return;
  }
  defer(cudaFree(d_b));

  if (PRINT_IF_ERROR(cudaMalloc((void**) &d_c, c.size() * sizeof(*c.data())))) {
    LOG(critical, "CUBLAS/{} device memory allocation failed for matrix C", IMPLEMENTATION_NAME);
    state.SkipWithError(
        fmt::format("CUBLAS/{} device memory allocation failed for matrix C", IMPLEMENTATION_NAME).c_str());
    return;
  }
  defer(cudaFree(d_c));

  if (PRINT_IF_ERROR(cublasSetMatrix(M, K, sizeof(*a.data()), a.data(), M, d_a, M))) {
    LOG(critical, "CUBLAS/{} setting of A matrix failed", IMPLEMENTATION_NAME);
    state.SkipWithError(fmt::format("CUBLAS/{} setting of A matrix failed", IMPLEMENTATION_NAME).c_str());
    return;
  }

  if (PRINT_IF_ERROR(cublasSetMatrix(K, N, sizeof(*b.data()), b.data(), K, d_b, K))) {
    LOG(critical, "CUBLAS/{} setting of B matrix failed", IMPLEMENTATION_NAME);
    state.SkipWithError(fmt::format("CUBLAS/{} setting of B matrix failed", IMPLEMENTATION_NAME).c_str());
    return;
  }

  if (PRINT_IF_ERROR(cublasSetMatrix(M, N, sizeof(*c.data()), c.data(), M, d_c, M))) {
    LOG(critical, "CUBLAS/{} setting of C matrix failed", IMPLEMENTATION_NAME);
    state.SkipWithError(fmt::format("CUBLAS/{} setting of C matrix failed", IMPLEMENTATION_NAME).c_str());
    return;
  }

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  for (auto _ : state) {
    cudaEventRecord(start, NULL);

    const cublasStatus_t cublas_err = cublas_gemm<device_type>(cublas_handle, CblasNoTrans, CblasNoTrans, M, N, K,
                                                               reinterpret_cast<device_type*>(&alpha), d_a, d_b,
                                                               reinterpret_cast<device_type*>(&beta), d_c);

    cudaEventRecord(stop, NULL);
    const auto cuda_err = cudaEventSynchronize(stop);

    state.PauseTiming();
    if (PRINT_IF_ERROR(cublas_err)) {
      state.SkipWithError(fmt::format("CUBLAS/{} failed to launch kernel", IMPLEMENTATION_NAME).c_str());
      break;
    }
    if (PRINT_IF_ERROR(cuda_err)) {
      state.SkipWithError(fmt::format("CUBLAS/{} failed to synchronize kernel", IMPLEMENTATION_NAME).c_str());
      break;
    }

    float msecTotal = 0.0f;
    if (PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop))) {
      state.SkipWithError(fmt::format("CUBLAS/{} failed to get elapsed time", IMPLEMENTATION_NAME).c_str());
      break;
    }
    state.SetIterationTime(msecTotal / 1000);
    state.ResumeTiming();
  }

  state.counters.insert(
      {{"M", M}, {"N", N}, {"K", K}, {"Flops", {2.0 * M * N * K, benchmark::Counter::kAvgThreadsRate}}});
  state.SetBytesProcessed(int64_t(state.iterations()) * a.size() * b.size() * c.size());
  state.SetItemsProcessed(int64_t(state.iterations()) * M * N * K);
}

static void CUBLAS_HGEMM(benchmark::State& state) {
  return CUBLAS<__half>(state);
}

static void CUBLAS_SGEMM(benchmark::State& state) {
  return CUBLAS<float>(state);
}

static void CUBLAS_DGEMM(benchmark::State& state) {
  return CUBLAS<double>(state);
}

static void CUBLAS_CGEMM(benchmark::State& state) {
  return CUBLAS<std::complex<float>>(state);
}

static void CUBLAS_ZGEMM(benchmark::State& state) {
  return CUBLAS<std::complex<double>>(state);
}

BENCHMARK(CUBLAS_HGEMM)->ALL_ARGS()->UseManualTime();
BENCHMARK(CUBLAS_SGEMM)->ALL_ARGS()->UseManualTime();
BENCHMARK(CUBLAS_DGEMM)->ALL_ARGS()->UseManualTime();
BENCHMARK(CUBLAS_CGEMM)->ALL_ARGS()->UseManualTime();
BENCHMARK(CUBLAS_ZGEMM)->ALL_ARGS()->UseManualTime();
