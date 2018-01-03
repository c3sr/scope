
#include <benchmark/benchmark.h>

#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <cuda_runtime.h>

#include "fmt/format.h"

#include "gemm/args.hpp"
#include "init/init.hpp"
#include "utils/utils.hpp"

enum class CUDA_BLAS_IMPLEMENTATION : int { BASIC = 1, TILED = 2 };

static std::string CUDA_BLAS_IMPLEMENTATION_STRING(const CUDA_BLAS_IMPLEMENTATION impl) {
  switch (impl) {
    case CUDA_BLAS_IMPLEMENTATION::BASIC:
      return "BASIC";
    case CUDA_BLAS_IMPLEMENTATION::TILED:
      return "TILED";
    default:
      return "UNDEFINED";
  }
}

template <int TILE_WIDTH>
__global__ void cuda_basic_matrix_multiply(float *A, float *B, float *C, int numARows, int numAColumns, int numBRows,
                                           int numBColumns, int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < numARows && col < numBColumns) {
    float sum = 0;
    for (int ii = 0; ii < numAColumns; ii++) {
      sum += A[row * numAColumns + ii] * B[ii * numBColumns + col];
    }
    C[row * numBColumns + col] = sum;
  }
}

template <int TILE_WIDTH>
__global__ void cuda_tiled_matrix_multiply(float *A, float *B, float *C, int numARows, int numAColumns, int numBRows,
                                           int numBColumns, int numCRows, int numCColumns) {
  __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
  __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y, Row = by * TILE_WIDTH + ty,
      Col      = bx * TILE_WIDTH + tx;
  float Pvalue = 0;

  for (int m = 0; m < (numAColumns - 1) / TILE_WIDTH + 1; ++m) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    if (Row < numARows && m * TILE_WIDTH + tx < numAColumns) {
      ds_M[ty][tx] = A[Row * numAColumns + m * TILE_WIDTH + tx];
    } else {
      ds_M[ty][tx] = 0;
    }
    if (Col < numBColumns && m * TILE_WIDTH + ty < numBRows) {
      ds_N[ty][tx] = B[(m * TILE_WIDTH + ty) * numBColumns + Col];
    } else {
      ds_N[ty][tx] = 0;
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix

#pragma unroll
    for (int k = 0; k < TILE_WIDTH; ++k) {
      Pvalue += ds_M[ty][k] * ds_N[k][tx];
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }
  if (Row < numCRows && Col < numCColumns) {
    C[Row * numCColumns + Col] = Pvalue;
  }
}

template <CUDA_BLAS_IMPLEMENTATION IMPLEMENTATION, int TILE_WIDTH>
static void CUDA_SGEMM(benchmark::State &state) {

  const std::string IMPLEMENTATION_NAME = CUDA_BLAS_IMPLEMENTATION_STRING(IMPLEMENTATION);

  if (!has_cuda) {
    state.SkipWithError(fmt::format("CUDA/SGEMM/{} no CUDA device found", IMPLEMENTATION_NAME).c_str());
    return;
  }

  const auto M     = state.range(0);
  const auto N     = state.range(1);
  const auto K     = state.range(2);
  const auto alpha = 1.0f;
  const auto beta  = 0.0f;

  (void) alpha;
  (void) beta;

  const auto numARows    = M;
  const auto numAColumns = K;
  const auto numBRows    = K;
  const auto numBColumns = N;
  const auto numCRows    = M;
  const auto numCColumns = N;

  (void) numARows;
  (void) numAColumns;
  (void) numBRows;
  (void) numBColumns;
  (void) numCRows;
  (void) numCColumns;

  dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 gridDim(ceil(((float) numBColumns) / blockDim.x), ceil(((float) numARows) / blockDim.y));

  if (gridDim.x >= cuda_device_prop.maxGridSize[0]) {
    const auto str = fmt::format("CUDA/SGEMM/{} the X grid dimension {} exceeds the max grid dimensions {}",
                                 IMPLEMENTATION_NAME, gridDim.x, cuda_device_prop.maxGridSize[0]);
    state.SkipWithError(str.c_str());
    return;
  }

  if (gridDim.x >= CUDA_MAX_GRID_SIZE) {
    const auto str = fmt::format("CUDA/SGEMM/{} the X grid dimension {} exceeds the max grid dimensions {}",
                                 IMPLEMENTATION_NAME, gridDim.x, CUDA_MAX_GRID_SIZE);
    state.SkipWithError(str.c_str());
    return;
  }

  if (gridDim.y >= cuda_device_prop.maxGridSize[1]) {
    const auto str = fmt::format("CUDA/SGEMM/{} the Y grid dimension {} exceeds the max grid dimensions {}",
                                 IMPLEMENTATION_NAME, gridDim.y, cuda_device_prop.maxGridSize[1]);
    state.SkipWithError(str.c_str());
    return;
  }

  if (gridDim.y >= CUDA_MAX_GRID_SIZE) {
    const auto str = fmt::format("CUDA/SGEMM/{} the Y grid dimension {} exceeds the max grid dimensions {}",
                                 IMPLEMENTATION_NAME, gridDim.y, CUDA_MAX_GRID_SIZE);
    state.SkipWithError(str.c_str());
    return;
  }

  auto a = std::vector<float>(M * K);
  auto b = std::vector<float>(K * N);
  auto c = std::vector<float>(M * N);

  std::iota(a.begin(), a.end(), 1);
  std::iota(b.begin(), b.end(), 1);
  std::fill(c.begin(), c.end(), 0);

  float *d_a{nullptr}, *d_b{nullptr}, *d_c{nullptr};

  if (PRINT_IF_ERROR(cudaMalloc((void **) &d_a, a.size() * sizeof(*a.data())))) {
    LOG(critical, "CUDA/SGEMM/{} device memory allocation failed for matrix A", IMPLEMENTATION_NAME);
    state.SkipWithError(
        fmt::format("CUDA/SGEMM/{} device memory allocation failed for matrix A", IMPLEMENTATION_NAME).c_str());
    return;
  }
  defer(cudaFree(d_a));

  if (PRINT_IF_ERROR(cudaMalloc((void **) &d_b, b.size() * sizeof(*b.data())))) {
    LOG(critical, "CUDA/SGEMM/{} device memory allocation failed for matrix B", IMPLEMENTATION_NAME);
    state.SkipWithError(
        fmt::format("CUDA/SGEMM/{} device memory allocation failed for matrix B", IMPLEMENTATION_NAME).c_str());
    return;
  }
  defer(cudaFree(d_b));

  if (PRINT_IF_ERROR(cudaMalloc((void **) &d_c, c.size() * sizeof(*c.data())))) {
    LOG(critical, "CUDA/SGEMM/{} device memory allocation failed for matrix C", IMPLEMENTATION_NAME);
    state.SkipWithError(
        fmt::format("CUDA/SGEMM/{} device memory allocation failed for matrix C", IMPLEMENTATION_NAME).c_str());
    return;
  }
  defer(cudaFree(d_c));

  if (PRINT_IF_ERROR(cudaMemcpy(d_a, a.data(), a.size() * sizeof(*a.data()), cudaMemcpyHostToDevice))) {
    state.SkipWithError(fmt::format("CUDA/SGEMM/{} failed to copy A Matrix to device", IMPLEMENTATION_NAME).c_str());
    return;
  }

  if (PRINT_IF_ERROR(cudaMemcpy(d_b, b.data(), b.size() * sizeof(*b.data()), cudaMemcpyHostToDevice))) {
    state.SkipWithError(fmt::format("CUDA/SGEMM/{} failed to copy B Matrix to device", IMPLEMENTATION_NAME).c_str());
    return;
  }

  if (PRINT_IF_ERROR(cudaMemcpy(d_c, c.data(), c.size() * sizeof(*c.data()), cudaMemcpyHostToDevice))) {
    state.SkipWithError(fmt::format("CUDA/SGEMM/{} failed to copy C Matrix to device", IMPLEMENTATION_NAME).c_str());
    return;
  }

#ifdef USE_CUDA_EVENTS
  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));
#endif //  USE_CUDA_EVENTS

  for (auto _ : state) {
#ifdef USE_CUDA_EVENTS
    cudaEventRecord(start, NULL);
#endif // USE_CUDA_EVENTS

    switch (IMPLEMENTATION) {
      case CUDA_BLAS_IMPLEMENTATION::BASIC:
        cuda_basic_matrix_multiply<TILE_WIDTH>
            <<<gridDim, blockDim>>>(d_a, d_b, d_c, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
        break;
      case CUDA_BLAS_IMPLEMENTATION::TILED:
        cuda_tiled_matrix_multiply<TILE_WIDTH>
            <<<gridDim, blockDim>>>(d_a, d_b, d_c, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
        break;
    }
#ifdef USE_CUDA_EVENTS
    cudaEventRecord(stop, NULL);
    const auto cuda_err = cudaEventSynchronize(stop);
#else  // USE_CUDA_EVENTS
    const auto cuda_err = cudaDeviceSynchronize();
#endif // USE_CUDA_EVENTS

    state.PauseTiming();
    if (PRINT_IF_ERROR(cuda_err)) {
      state.SkipWithError(fmt::format("CUDA/SGEMM/{} failed to synchronize", IMPLEMENTATION_NAME).c_str());
      break;
    }

#ifdef USE_CUDA_EVENTS
    float msecTotal = 0.0f;
    if (PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop))) {
      state.SkipWithError(fmt::format("CUDA/SGEMM/{} failed to get elapsed time", IMPLEMENTATION_NAME).c_str());
      break;
    }
    state.SetIterationTime(msecTotal / 1000);
#endif // USE_CUDA_EVENTS

    state.ResumeTiming();
  }

  state.counters.insert({{"M", M}, {"N", N}, {"K", K}, {"Flops", {2.0 * M * N * K, benchmark::Counter::kAvgThreadsRate}}, {"IMPLEMENTATION_TYPE", (int) IMPLEMENTATION}});
  if (IMPLEMENTATION != CUDA_BLAS_IMPLEMENTATION::BASIC) {
    state.counters.insert({{"TILE_WIDTH", TILE_WIDTH}});
  }
  state.SetBytesProcessed(int64_t(state.iterations()) * 2 * M * N * K);
}

static void CUDA_SGEMM_BASIC(benchmark::State &state) {
  constexpr auto TILE_WIDTH = 16; // this is not used
  CUDA_SGEMM<CUDA_BLAS_IMPLEMENTATION::BASIC, TILE_WIDTH>(state);
}

template <int TILE_WIDTH>
static void CUDA_SGEMM_TILED(benchmark::State &state) {
  CUDA_SGEMM<CUDA_BLAS_IMPLEMENTATION::TILED, TILE_WIDTH>(state);
}

#ifdef USE_CUDA_EVENTS
BENCHMARK(CUDA_SGEMM_BASIC)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(CUDA_SGEMM_TILED, 16)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(CUDA_SGEMM_TILED, 32)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(CUDA_SGEMM_TILED, 64)->ALL_ARGS()->UseManualTime();
#else  // USE_CUDA_EVENTS
BENCHMARK(CUDA_SGEMM_BASIC)->ALL_ARGS();
BENCHMARK_TEMPLATE(CUDA_SGEMM_TILED, 16)->ALL_ARGS();
BENCHMARK_TEMPLATE(CUDA_SGEMM_TILED, 32)->ALL_ARGS();
BENCHMARK_TEMPLATE(CUDA_SGEMM_TILED, 64)->ALL_ARGS();
#endif // USE_CUDA_EVENTS
