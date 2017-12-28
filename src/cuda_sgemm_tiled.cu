// System includes
#include <assert.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

template <int TILE_WIDTH>
__global__ void tiled_matrix_multiply(float *A, float *B, float *C,
                                      int numARows, int numAColumns,
                                      int numBRows, int numBColumns,
                                      int numCRows, int numCColumns) {
  __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
  __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y,
      Row = by * TILE_WIDTH + ty, Col = bx * TILE_WIDTH + tx;
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

template <int TILE_WIDTH>
static void CUDA_SGEMM_TILED(benchmark::State &state) {

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
    LOG(critical,
        "CUDA/SGEMM/TILED device memory allocation failed for matrix A");
    return;
  }
  defer(cudaFree(d_a));

  cuda_err = cudaMalloc((void **)&d_b, b.size() * sizeof(*b.data()));
  if (cuda_err != cudaSuccess) {
    LOG(critical,
        "CUDA/SGEMM/TILED device memory allocation failed for matrix B");
    return;
  }
  defer(cudaFree(d_b));

  cuda_err = cudaMalloc((void **)&d_c, c.size() * sizeof(*c.data()));
  if (cuda_err != cudaSuccess) {
    LOG(critical,
        "CUDA/SGEMM/TILED device memory allocation failed for matrix C");
    return;
  }
  defer(cudaFree(d_c));

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

  cudaEvent_t start, stop;
  CUDA_PERROR(cudaEventCreate(&start));
  CUDA_PERROR(cudaEventCreate(&stop));

  for (auto _ : state) {
    cudaEventRecord(start, NULL);

    tiled_matrix_multiply<TILE_WIDTH><<<gridDim, blockDim>>>(
        d_a, d_b, d_c, numARows, numAColumns, numBRows, numBColumns);

    cuda_err = cudaDeviceSynchronize();

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    state.PauseTiming();
    if (CUDA_PERROR(cuda_err) != cudaSuccess) {
      break;
    }

    float msecTotal = 0.0f;
    if (cuda_err = CUDA_PERROR(cudaEventElapsedTime(&msecTotal, start, stop))) {
      state.SkipWithError("CUDA/SGEMM/TILED failed to get elapsed time");
    }
    state.SetIterationTime(msecTotal * 1000);
    state.ResumeTiming();
  }

  state.counters.insert(
      {{"M", M}, {"N", N}, {"K", K}, {"TILE_WIDTH", TILE_WIDTH}});
  state.SetBytesProcessed(int64_t(state.iterations()) * 2 * M * N * K);
}

BENCHMARK_TEMPLATE(CUDA_SGEMM_TILED, 16)->SGEMM_ARGS();
BENCHMARK_TEMPLATE(CUDA_SGEMM_TILED, 32)->SGEMM_ARGS();
BENCHMARK_TEMPLATE(CUDA_SGEMM_TILED, 64)->SGEMM_ARGS();
