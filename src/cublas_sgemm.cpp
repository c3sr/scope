#include <benchmark/benchmark.h>

#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "init.hpp"
#include "utils.hpp"
#include "utils_sgemm.hpp"

static void CUBLAS_SGEMM(benchmark::State &state) {
  const auto M     = state.range(0);
  const auto N     = state.range(1);
  const auto K     = state.range(2);
  const auto alpha = 1.0f;
  const auto beta  = 0.0f;

  auto a = std::vector<float>(M * K);
  auto b = std::vector<float>(K * N);
  auto c = std::vector<float>(M * N);

  std::iota(a.begin(), a.end(), 1);
  std::iota(b.begin(), b.end(), 1);
  std::fill(c.begin(), c.end(), 0);

  cublasHandle_t cublas_handle;

  auto cublas_err = cublasCreate(&cublas_handle);
  if (cublas_err != CUBLAS_STATUS_SUCCESS) {
    LOG(critical, "CUBLAS/SGEMM initialization failed");
    return;
  }
  defer(cublasDestroy(cublas_handle));

  float *d_a{nullptr}, *d_b{nullptr}, *d_c{nullptr};

  auto cuda_err = cudaMalloc((void **) &d_a, a.size() * sizeof(*a.data()));
  if (cuda_err != cudaSuccess) {
    LOG(critical, "CUBLAS/SGEMM device memory allocation failed for matrix A");
    return;
  }
  defer(cudaFree(d_a));

  cuda_err = cudaMalloc((void **) &d_b, b.size() * sizeof(*b.data()));
  if (cuda_err != cudaSuccess) {
    LOG(critical, "CUBLAS/SGEMM device memory allocation failed for matrix B");
    return;
  }
  defer(cudaFree(d_b));

  cuda_err = cudaMalloc((void **) &d_c, c.size() * sizeof(*c.data()));
  if (cuda_err != cudaSuccess) {
    LOG(critical, "CUBLAS/SGEMM device memory allocation failed for matrix C");
    return;
  }
  defer(cudaFree(d_c));

  cublas_err = cublasSetMatrix(M, N, sizeof(*a.data()), a.data(), M, d_a, M);
  if (cublas_err != CUBLAS_STATUS_SUCCESS) {
    LOG(critical, "CUBLAS/SGEMM setting of A matrix failed");
    return;
  }

  cublas_err = cublasSetMatrix(K, N, sizeof(*b.data()), b.data(), K, d_b, K);
  if (cublas_err != CUBLAS_STATUS_SUCCESS) {
    LOG(critical, "CUBLAS/SGEMM setting of B matrix failed");
    return;
  }

  cublas_err = cublasSetMatrix(M, K, sizeof(*c.data()), c.data(), M, d_c, M);
  if (cublas_err != CUBLAS_STATUS_SUCCESS) {
    LOG(critical, "CUBLAS/SGEMM setting of C matrix failed");
    return;
  }

  for (auto _ : state) {
    // Use the fact that C^T = (B^T . A^T)^T for optimization
    const auto cublas_err =
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, M, d_a, K, &beta, d_c, N);
    state.PauseTiming();
    if (cublas_err != CUBLAS_STATUS_SUCCESS) {
      LOG(critical, "CUBLAS/SGEMM operation failed");
      break;
    }
    state.ResumeTiming();
  }

  state.counters.insert({{"M", M}, {"N", N}, {"K", K}});
  state.SetBytesProcessed(int64_t(state.iterations()) * 2 * M * N * K);
  state.SetItemsProcessed(int64_t(state.iterations()) * 2 * M * N * K);
}

BENCHMARK(CUBLAS_SGEMM)->SGEMM_ARGS();
