
#include <benchmark/benchmark.h>

#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <cuda_runtime.h>

#include "init.hpp"
#include "utils.hpp"
#include "utils_cuda.hpp"
#include "utils_vectoradd.hpp"

template <typename T, int COARSINING_FACTOR = 1, int BLOCK_SIZE = 1>
__global__ void cuda_vector_add(T *in1, T *in2, T *out, size_t len) {
  // todo: implement COARSINING_FACTOR
  int index = threadIdx.x + blockIdx.x * BLOCK_SIZE;
  if (index < len) {
    out[index] = in1[index] + in2[index];
  }
}

template <typename T, int COARSINING_FACTOR = 1, int BLOCK_SIZE = 128>
static void CUDA_VECTOR_ADD(benchmark::State &state) {

  const size_t N = state.range(0);

  auto a = std::vector<T>(N);
  auto b = std::vector<T>(N);
  auto c = std::vector<T>(N);

  std::fill(a.begin(), a.end(), 1);
  std::fill(b.begin(), b.end(), 1);
  std::fill(c.begin(), c.end(), 0);

  T *d_a{nullptr}, *d_b{nullptr}, *d_c{nullptr};

  auto cuda_err = cudaMalloc((void **)&d_a, a.size() * sizeof(*a.data()));
  if (cuda_err != cudaSuccess) {
    LOG(critical,
        "CUDA/VECTOR_ADD/BASIC device memory allocation failed for vector A");
    return;
  }
  defer(cudaFree(d_a));

  cuda_err = cudaMalloc((void **)&d_b, b.size() * sizeof(*b.data()));
  if (cuda_err != cudaSuccess) {
    LOG(critical,
        "CUDA/VECTOR_ADD/BASIC device memory allocation failed for vector B");
    return;
  }
  defer(cudaFree(d_b));

  cuda_err = cudaMalloc((void **)&d_c, c.size() * sizeof(*c.data()));
  if (cuda_err != cudaSuccess) {
    LOG(critical,
        "CUDA/VECTOR_ADD/BASIC device memory allocation failed for vector C");
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

  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim(ceil(((float)N) / blockDim.x));

  cudaEvent_t start, stop;
  CUDA_PERROR(cudaEventCreate(&start));
  CUDA_PERROR(cudaEventCreate(&stop));

  for (auto _ : state) {
    cudaEventRecord(start, NULL);

    cuda_vector_add<T, COARSINING_FACTOR, BLOCK_SIZE>
        <<<gridDim, blockDim>>>(d_a, d_b, d_c, N);

    cuda_err = cudaDeviceSynchronize();

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    state.PauseTiming();
    if (CUDA_PERROR(cuda_err) != cudaSuccess) {
      break;
    }

    float msecTotal = 0.0f;
    if (cuda_err = CUDA_PERROR(cudaEventElapsedTime(&msecTotal, start, stop))) {
      state.SkipWithError("CUDA/VECTOR_ADD/BASIC failed to get elapsed time");
    }
    state.SetIterationTime(msecTotal / 1000);
    state.ResumeTiming();
  }

  state.counters.insert({{"N", N},
                         {"BLOCK_SIZE", BLOCK_SIZE},
                         {"COARSINING_FACTOR", COARSINING_FACTOR}});
  state.SetBytesProcessed(int64_t(state.iterations()) * N);
}

BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, char, 1, 32)
    ->VECTORADD_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, int, 1, 32)
    ->VECTORADD_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, float, 1, 32)
    ->VECTORADD_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, double, 1, 32)
    ->VECTORADD_ARGS()
    ->UseManualTime();

BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, char, 1, 64)
    ->VECTORADD_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, int, 1, 64)
    ->VECTORADD_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, float, 1, 64)
    ->VECTORADD_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, double, 1, 64)
    ->VECTORADD_ARGS()
    ->UseManualTime();

BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, char, 1, 128)
    ->VECTORADD_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, int, 1, 128)
    ->VECTORADD_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, float, 1, 128)
    ->VECTORADD_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, double, 1, 128)
    ->VECTORADD_ARGS()
    ->UseManualTime();

BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, char, 1, 32)->VECTORADD_ARGS();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, int, 1, 32)->VECTORADD_ARGS();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, float, 1, 32)->VECTORADD_ARGS();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, double, 1, 32)->VECTORADD_ARGS();

BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, char, 1, 64)->VECTORADD_ARGS();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, int, 1, 64)->VECTORADD_ARGS();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, float, 1, 64)->VECTORADD_ARGS();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, double, 1, 64)->VECTORADD_ARGS();

BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, char, 1, 128)->VECTORADD_ARGS();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, int, 1, 128)->VECTORADD_ARGS();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, float, 1, 128)->VECTORADD_ARGS();
BENCHMARK_TEMPLATE(CUDA_VECTOR_ADD, double, 1, 128)->VECTORADD_ARGS();
