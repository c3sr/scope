#include <benchmark/benchmark.h>

#ifdef ENABLE_CUTLASS

#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <cblas.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#if CUDA_VERSION < 9000
// CUDA 9.0 introduces a new, light-weight barrier synchronization primitive
// that operates at the warp-scope. This is required to ensure visibility of
// reads/writes among threads that can make indepenent progress on Volta.
// For previous CUDA versions these synchronizations not necessary, and we
// define an empty function as a convenience for backward compatibility.
#ifndef __syncwarp
#define __syncwarp(...)
#endif // __syncwarp
#endif // CUDA_VERSION < 9000

#if 0
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100 4101 4181 4211 4244 4273 4324 4503 4512 4522 4700 4714 4717 4800)
#elif defined __INTEL_COMPILER
#pragma warning push
#pragma warning disable 2196 279 1684 2259
#elif defined __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wall"
#pragma clang diagnostic ignored "-Wextra"
#pragma clang diagnostic ignored "-Wunused"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-variable"
#elif defined __GNUC__ && __GNUC__ >= 5
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wunused"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif
#endif

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

template <typename T, cutlass::gemm::tiling_strategy::kind_t tiling_strategy>
static cudaError_t cutlass_gemm(int M, int N, int K, T* alpha, T* A, T* B, T* beta, T* C) {
  using namespace cutlass;
  using namespace cutlass::gemm;

  using value_t = T;
  using accum_t = T;

  constexpr auto accumulator_alignment = sizeof(accum_t);
  constexpr auto operator_alignment    = accumulator_alignment;
  constexpr auto math_op               = math_operation_class_t::scalar;

  constexpr auto TransformA = matrix_transform_t::Transpose;
  constexpr auto TransformB = matrix_transform_t::Transpose;

  // Define the epilogue functor
  using epilogue_op_t = blas_scaled_epilogue<T, T, T>;

  const epilogue_op_t epilogue_op(*alpha, *beta);

  const auto conf = cutlass::gemm::device_gemm<tiling_strategy,    //< Tile-sizing classification
                                               math_op,            //< Indicates which class of math operation to select
                                               TransformA,         //< Transformation op for matrix A
                                               operator_alignment, //< Alignment (in bytes) of A operand
                                               TransformB,         //< Transformation op for matrix B
                                               operator_alignment, //< Alignment (in bytes) of B operand
                                               value_t,            //< Multiplicand value type (matrices A and B)
                                               accum_t,            //< Accumulator value type (matrix C and scalars)
                                               epilogue_op_t,      //< Epilogue operation to update matrix C
                                               accumulator_alignment //< Alignment (in bytes) of C operand
                                               >(M, N, K, epilogue_op, B, A, C);

  return conf.result;
}

template <typename T, cutlass::gemm::tiling_strategy::kind_t tiling_strategy>
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

    const auto cutlass_err = cutlass_gemm<device_type, tiling_strategy>(
        M, N, K, reinterpret_cast<device_type*>(&alpha), d_a, d_b, reinterpret_cast<device_type*>(&beta), d_c);

    cudaEventRecord(stop, NULL);
    const auto cuda_err = cudaEventSynchronize(stop);

    state.PauseTiming();
    if (PRINT_IF_ERROR(cutlass_err)) {
      state.SkipWithError(fmt::format("CUTLASS/{} failed to launch kernel", IMPLEMENTATION_NAME).c_str());
      break;
    }
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

template <cutlass::gemm::tiling_strategy::kind_t tiling_strategy>
static void CUTLASS_HGEMM(benchmark::State& state) {
  return CUTLASS<__half, tiling_strategy>(state);
}

template <cutlass::gemm::tiling_strategy::kind_t tiling_strategy>
static void CUTLASS_SGEMM(benchmark::State& state) {
  return CUTLASS<float, tiling_strategy>(state);
}
template <cutlass::gemm::tiling_strategy::kind_t tiling_strategy>
static void CUTLASS_DGEMM(benchmark::State& state) {
  return CUTLASS<double, tiling_strategy>(state);
}
template <cutlass::gemm::tiling_strategy::kind_t tiling_strategy>
static void CUTLASS_I8GEMM(benchmark::State& state) {
  return CUTLASS<int8_t, tiling_strategy>(state);
}
template <cutlass::gemm::tiling_strategy::kind_t tiling_strategy>
static void CUTLASS_I32GEMM(benchmark::State& state) {
  return CUTLASS<int32_t, tiling_strategy>(state);
}

#define BENCHMARK_SMALL_TILING(b)                                                                                      \
  BENCHMARK_TEMPLATE(b, cutlass::gemm::tiling_strategy::Small)                                                         \
      ->Args({256, 169, 1})                                                                                            \
      ->Args({256, 729, 1})                                                                                            \
      ->Args({384, 169, 1})                                                                                            \
      ->Args({128, 169, 129})

#define BENCHMARK_MEDIUM_TILING(b)                                                                                     \
  BENCHMARK_TEMPLATE(b, cutlass::gemm::tiling_strategy::Large)                                                         \
      ->Args({1000, 1, 1})                                                                                             \
      ->Args({50, 1000, 1})                                                                                            \
      ->Args({512, 2, 512})                                                                                            \
      ->Args({1024, 2, 512})                                                                                           \
      ->Args({512, 4, 512})                                                                                            \
      ->Args({1024, 4, 512})

#define BENCHMARK_LARGE_TILING(b)                                                                                      \
  BENCHMARK_TEMPLATE(b, cutlass::gemm::tiling_strategy::Large)                                                         \
      ->Args({384, 169, 2304})                                                                                         \
      ->Args({50, 1000, 4096})                                                                                         \
      ->Args({50, 4096, 1})                                                                                            \
      ->Args({96, 3025, 1})                                                                                            \
      ->Args({96, 3025, 363})                                                                                          \
      ->Args({35, 700, 2048})                                                                                          \
      ->Args({1024, 700, 512})

#define BENCHMARK_HUGE_TILING(b)                                                                                       \
  BENCHMARK_TEMPLATE(b, cutlass::gemm::tiling_strategy::Huge)                                                          \
      ->Args({128, 169, 1728})                                                                                         \
      ->Args({128, 729, 1200})                                                                                         \
      ->Args({192, 169, 1728})                                                                                         \
      ->Args({50, 4096, 4096})                                                                                         \
      ->Args({50, 4096, 9216})                                                                                         \
      ->Args({5124, 700, 2048})                                                                                        \
      ->Args({5124, 700, 2560})                                                                                        \
      ->Args({7680, 1, 2560})                                                                                          \
      ->Args({7680, 2, 2560})                                                                                          \
      ->Args({7680, 4, 2560})                                                                                          \
      ->Args({3072, 1, 1024})                                                                                          \
      ->Args({3072, 2, 1024})                                                                                          \
      ->Args({3072, 4, 1024})                                                                                          \
      ->Args({7680, 1500, 2560})                                                                                       \
      ->Args({6144, 1500, 2048})                                                                                       \
      ->Args({4608, 1500, 1536})                                                                                       \
      ->Args({8448, 1500, 2816})                                                                                       \
      ->Args({3072, 1500, 1024})                                                                                       \
      ->Args({7680, 3000, 2560})                                                                                       \
      ->Args({6144, 3000, 2048})                                                                                       \
      ->Args({4608, 3000, 1536})                                                                                       \
      ->Args({8448, 3000, 2816})                                                                                       \
      ->Args({3072, 3000, 1024})                                                                                       \
      ->Args({7680, 6000, 2560})                                                                                       \
      ->Args({6144, 6000, 2048})                                                                                       \
      ->Args({4608, 6000, 1536})                                                                                       \
      ->Args({8448, 6000, 2816})                                                                                       \
      ->Args({3072, 6000, 1024})

#define BENCHMARK_WIDE_TILING(b)                                                                                       \
  BENCHMARK_TEMPLATE(b, cutlass::gemm::tiling_strategy::Wide)                                                          \
      ->Args({128, 169, 1728})                                                                                         \
      ->Args({128, 729, 1200})                                                                                         \
      ->Args({192, 169, 1728})

#define BENCHMARK_TALL_TILING(b)                                                                                       \
  BENCHMARK_TEMPLATE(b, cutlass::gemm::tiling_strategy::Tall)                                                          \
      ->Args({512, 1, 500000})                                                                                         \
      ->Args({1024, 1, 500000})                                                                                        \
      ->Args({512, 2, 500000})                                                                                         \
      ->Args({1024, 2, 500000})                                                                                        \
      ->Args({512, 4, 500000})                                                                                         \
      ->Args({1024, 4, 500000})

#define BENCHMARK_CUTLASS(b)                                                                                           \
  BENCHMARK_SMALL_TILING(b)->UseManualTime();                                                                          \
  BENCHMARK_MEDIUM_TILING(b)->UseManualTime();

#if 0
  BENCHMARK_LARGE_TILING(b)->UseManualTime();                                                                          \
  BENCHMARK_HUGE_TILING(b)->UseManualTime();                                                                           \
  BENCHMARK_WIDE_TILING(b)->UseManualTime();                                                                           \
  BENCHMARK_TALL_TILING(b)->UseManualTime()
#endif

// BENCHMARK_CUTLASS(CUTLASS_HGEMM);
BENCHMARK_CUTLASS(CUTLASS_SGEMM);
BENCHMARK_CUTLASS(CUTLASS_DGEMM);
// BENCHMARK_CUTLASS(CUTLASS_I32GEMM);
// BENCHMARK_CUTLASS(CUTLASS_I8GEMM);

#if 0
#ifdef _MSC_VER
#pragma warning(pop)
#elif defined __INTEL_COMPILER
#pragma warning pop
#elif defined __clang__
#pragma clang diagnostic pop
#elif defined __GNUC__ && __GNUC__ >= 5
#pragma GCC diagnostic pop
#endif
#endif

#endif // ENABLE_CUTLASS
