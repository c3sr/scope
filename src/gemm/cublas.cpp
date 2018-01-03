#include <benchmark/benchmark.h>

#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "init/init.hpp"
#include "utils/utils.hpp"

#include "gemm/args.hpp"
#include "gemm/utils.hpp"

template <typename T>
static void CUBLAS(benchmark::State &state) {
  static const std::string IMPLEMENTATION_NAME = gemm::detail::implementation_name<T>();

  if (!has_cuda) {
    state.SkipWithError("CUDA/SGEMM no CUDA device found");
    return;
  }

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
  std::fill(c.begin(), c.end(), 0);

  cublasHandle_t cublas_handle;

  if (PRINT_IF_ERROR(cublasCreate(&cublas_handle))) {
    LOG(critical, "CUBLAS/{} initialization failed", IMPLEMENTATION_NAME);
    state.SkipWithError(fmt::format("CUBLAS/{} initialization failed", IMPLEMENTATION_NAME).c_str());
    return;
  }
  defer(cublasDestroy(cublas_handle));

  using device_type = typename gemm::detail::cuda_type<T>::type;

  device_type *d_a{nullptr}, *d_b{nullptr}, *d_c{nullptr};

  if (PRINT_IF_ERROR(cudaMalloc((void **) &d_a, a.size() * sizeof(*a.data())))) {
    LOG(critical, "CUBLAS/{} device memory allocation failed for matrix A", IMPLEMENTATION_NAME);
    state.SkipWithError(
        fmt::format("CUBLAS/{} device memory allocation failed for matrix A", IMPLEMENTATION_NAME).c_str());
    return;
  }
  defer(cudaFree(d_a));

  if (PRINT_IF_ERROR(cudaMalloc((void **) &d_b, b.size() * sizeof(*b.data())))) {
    LOG(critical, "CUBLAS/{} device memory allocation failed for matrix B", IMPLEMENTATION_NAME);
    state.SkipWithError(
        fmt::format("CUBLAS/{} device memory allocation failed for matrix B", IMPLEMENTATION_NAME).c_str());
    return;
  }
  defer(cudaFree(d_b));

  if (PRINT_IF_ERROR(cudaMalloc((void **) &d_c, c.size() * sizeof(*c.data())))) {
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

    cublasStatus_t cublas_err;

    // Use the fact that C^T = (B^T . A^T)^T for optimization
    if constexpr (std::is_same<T, float>::value) {
      cublas_err = cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, M, d_a, K, &beta, d_c, N);
    } else if constexpr (std::is_same<T, double>::value) {
      cublas_err = cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, M, d_a, K, &beta, d_c, N);
    } else if constexpr (std::is_same<T, std::complex<float>>::value) {
      cublas_err =
          cublasCgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, reinterpret_cast<device_type *>(&alpha), d_b, M,
                      d_a, K, reinterpret_cast<device_type *>(&beta), d_c, N);
    } else if constexpr (std::is_same<T, std::complex<double>>::value) {
      cublas_err =
          cublasZgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, reinterpret_cast<device_type *>(&alpha), d_b, M,
                      d_a, K, reinterpret_cast<device_type *>(&beta), d_c, N);
    }

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
  state.SetLabel(IMPLEMENTATION_NAME);
  state.SetBytesProcessed(int64_t(state.iterations()) * a.size() * b.size() * c.size());
  state.SetItemsProcessed(int64_t(state.iterations()) * M * N * K);
}

static void CUBLAS_SGEMM(benchmark::State &state) {
  return CUBLAS<float>(state);
}

static void CUBLAS_DGEMM(benchmark::State &state) {
  return CUBLAS<double>(state);
}

static void CUBLAS_CGEMM(benchmark::State &state) {
  return CUBLAS<std::complex<float>>(state);
}

static void CUBLAS_ZGEMM(benchmark::State &state) {
  return CUBLAS<std::complex<double>>(state);
}

BENCHMARK(CUBLAS_SGEMM)->ALL_ARGS()->UseManualTime();
BENCHMARK(CUBLAS_DGEMM)->ALL_ARGS()->UseManualTime();
BENCHMARK(CUBLAS_CGEMM)->ALL_ARGS()->UseManualTime();
BENCHMARK(CUBLAS_ZGEMM)->ALL_ARGS()->UseManualTime();

