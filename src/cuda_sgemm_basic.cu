
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
#include "utils_cuda.hpp"
#include "utils_sgemm.hpp"

template <int TILE_WIDTH>
__global__ void basic_matrix_multiply(float *A, float *B, float *C,
                                      int numARows, int numAColumns,
                                      int numBRows, int numBColumns) {
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

static void CUDA_SGEMM_BASIC(benchmark::State &state) {
  static constexpr int TILE_WIDTH = 16;

  const auto M = state.range(0);
  const auto N = state.range(1);
  const auto K = state.range(2);
  const auto alpha = 1.0f;
  const auto beta = 0.0f;

  (void)alpha;
  (void)beta;

  const auto numARows = M;
  const auto numAColumns = K;
  const auto numBRows = K;
  const auto numBColumns = N;
  const auto numCRows = M;
  const auto numCColumns = N;

  (void)numARows;
  (void)numAColumns;
  (void)numBRows;
  (void)numBColumns;
  (void)numCRows;
  (void)numCColumns;

  auto a = std::vector<float>(M * K);
  auto b = std::vector<float>(K * N);
  auto c = std::vector<float>(M * N);

  std::iota(a.begin(), a.end(), 1);
  std::iota(b.begin(), b.end(), 1);
  std::fill(c.begin(), c.end(), 0);

  float *d_a{nullptr}, *d_b{nullptr}, *d_c{nullptr};

  auto cuda_err = cudaMalloc((void **)&d_a, a.size() * sizeof(*a.data()));
  if (cuda_err != cudaSuccess) {
    LOG(critical, "CUBLAS/SGEMM device memory allocation failed for matrix A");
    return;
  }
  make_defer([&]() { cudaFree(d_a); });

  cuda_err = cudaMalloc((void **)&d_b, b.size() * sizeof(*b.data()));
  if (cuda_err != cudaSuccess) {
    LOG(critical, "CUBLAS/SGEMM device memory allocation failed for matrix B");
    return;
  }
  make_defer([&]() { cudaFree(d_b); });

  cuda_err = cudaMalloc((void **)&d_c, c.size() * sizeof(*c.data()));
  if (cuda_err != cudaSuccess) {
    LOG(critical, "CUBLAS/SGEMM device memory allocation failed for matrix C");
    return;
  }
  make_defer([&]() { cudaFree(d_c); });

  cuda_err = CUDA_PERROR(cudaMemcpy(d_a, a.data(), a.size() * sizeof(*a.data()),
                                    cudaMemcpyHostToDevice));
  if (cuda_err != cudaSuccess) {
    return;
  }

  cuda_err = CUDA_PERROR(cudaMemcpy(d_b, b.data(), b.size() * sizeof(*b.data()),
                                    cudaMemcpyHostToDevice));
  if (cuda_err != cudaSuccess) {
    return;
  }

  cuda_err = CUDA_PERROR(cudaMemcpy(d_c, c.data(), c.size() * sizeof(*c.data()),
                                    cudaMemcpyHostToDevice));
  if (cuda_err != cudaSuccess) {
    return;
  }

  dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 gridDim(ceil(((float)numBColumns) / blockDim.x),
               ceil(((float)numARows) / blockDim.y));

  for (auto _ : state) {
    basic_matrix_multiply<TILE_WIDTH><<<gridDim, blockDim>>>(
        d_a, d_b, d_c, numARows, numAColumns, numBRows, numBColumns);
    cuda_err = cudaDeviceSynchronize();

    state.PauseTiming();
    if (CUDA_PERROR(cuda_err) != cudaSuccess) {
      break;
    }
    state.ResumeTiming();
  }

  state.counters.insert({{"M", M}, {"N", N}, {"K", K}});
}

BENCHMARK(CUDA_SGEMM_BASIC)->SGEMM_ARGS();
