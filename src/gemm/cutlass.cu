#include <benchmark/benchmark.h>

#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <cblas.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// Cutlass GEMM API
#include <cutlass/gemm/dispatch.h>
#include <cutlass/gemm/epilogue_function.h>
#include <cutlass/util/util.h>

#ifdef PRINT_IF_ERROR
#undef PRINT_IF_ERROR
#endif // PRINT_IF_ERROR

#include "init/init.hpp"
#include "utils/utils.hpp"

#include "gemm/args.hpp"
#include "gemm/utils.hpp"

// Mask for all 32 threads in a warp.
#define CUDA_WARP_ALL 0xFFFFFFFF

#if defined(CUDA_VERSION) && CUDA_VERSION < 9000
// CUDA 9.0 introduces a new, light-weight barrier synchronization primitive
// that operates at the warp-scope. This is required to ensure visibility of
// reads/writes among threads that can make indepenent progress on Volta.
// For previous CUDA versions these synchronizations not necessary, and we
// define an empty function as a convenience for backward compatibility.
__device__ inline void __syncwarp(unsigned mask = CUDA_WARP_ALL) {
}

#endif

template <typename T>
static cudaError_t cutlass_gemm(int M, int N, int K, T* alpha, T* A, T* B, T* beta, T* C) {
  using namespace cutlass;
  using namespace cutlass::gemm;

  using value_t          = T;
  using accum_t          = T;
  constexpr auto math_op = math_operation_class_t::scalar;

  constexpr auto accumulator_alignment = sizeof(accum_t);
  constexpr auto operator_alignment    = accumulator_alignment;

  constexpr auto TransformA = matrix_transform_t::NonTranspose;
  constexpr auto TransformB = matrix_transform_t::NonTranspose;

  // using block_task_policy_t = gemm_policy<T, T, TransformA, TransformB, tiling_strategy::Medium>;
  // Define the epilogue functor
  using epilogue_op_t = blas_scaled_epilogue<T, T, T>;

  const epilogue_op_t epilogue_op(*alpha, *beta);

  const auto conf = cutlass::gemm::device_gemm<tiling_strategy::Medium, ///< Tile-sizing classification
                                               math_op,    ///< Indicates which class of math operation to select
                                               TransformA, ///< Transformation op for matrix A
                                               operator_alignment,   ///< Alignment (in bytes) of A operand
                                               TransformB,           ///< Transformation op for matrix B
                                               operator_alignment,   ///< Alignment (in bytes) of B operand
                                               value_t,              ///< Multiplicand value type (matrices A and B)
                                               accum_t,              ///< Accumulator value type (matrix C and scalars)
                                               epilogue_op_t,        ///< Epilogue operation to update matrix C
                                               accumulator_alignment ///< Alignment (in bytes) of C operand
                                               >(M, N, K, epilogue_op, A, B, C);

  return conf.result;
}

template <typename T>
static void CUTLASS(benchmark::State& state) {
  static const std::string IMPLEMENTATION_NAME = gemm::detail::implementation_name<T>();
  state.SetLabel(fmt::format("CUTLASS/{}", IMPLEMENTATION_NAME));

  if (!has_cuda) {
    state.SkipWithError("CUDA/SGEMM no CUDA device found");
    return;
  }

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

  using device_type = typename gemm::detail::cuda_type<T>::type;

  device_type *d_a{nullptr}, *d_b{nullptr}, *d_c{nullptr};

  if (PRINT_IF_ERROR(cudaMalloc((void**) &d_a, a.size() * sizeof(*a.data())))) {
    LOG(critical, "CUTLASS/{} device memory allocation failed for matrix A", IMPLEMENTATION_NAME);
    state.SkipWithError(
        fmt::format("CUTLASS/{} device memory allocation failed for matrix A", IMPLEMENTATION_NAME).c_str());
    return;
  }
  defer(cudaFree(d_a));

  if (PRINT_IF_ERROR(cudaMalloc((void**) &d_b, b.size() * sizeof(*b.data())))) {
    LOG(critical, "CUTLASS/{} device memory allocation failed for matrix B", IMPLEMENTATION_NAME);
    state.SkipWithError(
        fmt::format("CUTLASS/{} device memory allocation failed for matrix B", IMPLEMENTATION_NAME).c_str());
    return;
  }
  defer(cudaFree(d_b));

  if (PRINT_IF_ERROR(cudaMalloc((void**) &d_c, c.size() * sizeof(*c.data())))) {
    LOG(critical, "CUTLASS/{} device memory allocation failed for matrix C", IMPLEMENTATION_NAME);
    state.SkipWithError(
        fmt::format("CUTLASS/{} device memory allocation failed for matrix C", IMPLEMENTATION_NAME).c_str());
    return;
  }
  defer(cudaFree(d_c));

  if (PRINT_IF_ERROR(cublasSetMatrix(M, K, sizeof(*a.data()), a.data(), M, d_a, M))) {
    LOG(critical, "CUTLASS/{} setting of A matrix failed", IMPLEMENTATION_NAME);
    state.SkipWithError(fmt::format("CUTLASS/{} setting of A matrix failed", IMPLEMENTATION_NAME).c_str());
    return;
  }

  if (PRINT_IF_ERROR(cublasSetMatrix(K, N, sizeof(*b.data()), b.data(), K, d_b, K))) {
    LOG(critical, "CUTLASS/{} setting of B matrix failed", IMPLEMENTATION_NAME);
    state.SkipWithError(fmt::format("CUTLASS/{} setting of B matrix failed", IMPLEMENTATION_NAME).c_str());
    return;
  }

  if (PRINT_IF_ERROR(cublasSetMatrix(M, N, sizeof(*c.data()), c.data(), M, d_c, M))) {
    LOG(critical, "CUTLASS/{} setting of C matrix failed", IMPLEMENTATION_NAME);
    state.SkipWithError(fmt::format("CUTLASS/{} setting of C matrix failed", IMPLEMENTATION_NAME).c_str());
    return;
  }

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  for (auto _ : state) {
    cudaEventRecord(start, NULL);

    cutlass_gemm<device_type>(M, N, K, reinterpret_cast<device_type*>(&alpha), d_a, d_b,
                              reinterpret_cast<device_type*>(&beta), d_c);

    cudaEventRecord(stop, NULL);
    const auto cuda_err = cudaEventSynchronize(stop);

    state.PauseTiming();
    if (PRINT_IF_ERROR(cuda_err)) {
      state.SkipWithError(fmt::format("CUTLASS/{} failed to synchronize kernel", IMPLEMENTATION_NAME).c_str());
      break;
    }

    float msecTotal = 0.0f;
    if (PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop))) {
      state.SkipWithError(fmt::format("CUTLASS/{} failed to get elapsed time", IMPLEMENTATION_NAME).c_str());
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

// static void CUTLASS_HGEMM(benchmark::State& state) {
//   return CUTLASS<__half>(state);
// }

static void CUTLASS_SGEMM(benchmark::State& state) {
  return CUTLASS<float>(state);
}

// static void CUTLASS_DGEMM(benchmark::State& state) {
//   return CUTLASS<double>(state);
// }

// static void CUTLASS_CGEMM(benchmark::State& state) {
//   return CUTLASS<std::complex<float>>(state);
// }

// static void CUTLASS_ZGEMM(benchmark::State& state) {
//   return CUTLASS<std::complex<double>>(state);
// }

// BENCHMARK(CUTLASS_HGEMM)->ALL_ARGS()->UseManualTime();
BENCHMARK(CUTLASS_SGEMM)->ALL_ARGS()->UseManualTime();
// BENCHMARK(CUTLASS_DGEMM)->ALL_ARGS()->UseManualTime();
// BENCHMARK(CUTLASS_CGEMM)->ALL_ARGS()->UseManualTime();
// BENCHMARK(CUTLASS_ZGEMM)->ALL_ARGS()->UseManualTime();
