
#include <benchmark/benchmark.h>

#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <cuda_runtime.h>

#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_store.cuh>

#include "fmt/format.h"

#include "init.hpp"
#include "utils.hpp"
#include "utils_cuda.hpp"

using namespace cub;

enum class CUDA_REDUCE_IMPLEMENTATION : int {
  BASIC                   = 999,
  RAKING_COMMUTATIVE_ONLY = BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
  RAKING                  = BLOCK_REDUCE_RAKING,
  WARP_REDUCTIONS         = BLOCK_REDUCE_WARP_REDUCTIONS,
};

template <typename T,
          int BLOCK_THREADS,
          int ITEMS_PER_THREAD,
          BlockReduceAlgorithm ALGORITHM>
__global__ void cuda_reduce_cub_kernel(T *d_in,      // Tile of input
                                       T *d_out,     // Tile aggregate
                                       size_t len) { // length of the input
  // Specialize BlockReduce type for our thread block
  typedef BlockReduce<T, BLOCK_THREADS, ALGORITHM> BlockReduceT;
  // Shared memory
  __shared__ typename BlockReduceT::TempStorage temp_storage;
  // Per-thread tile data
  int data[ITEMS_PER_THREAD];
  LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, d_in, data);
  // Compute sum
  int aggregate = BlockReduceT(temp_storage).Sum(data);
  // Store aggregate and elapsed clocks
  if (threadIdx.x == 0) {
    *d_out = aggregate;
  }
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD = 1>
__global__ void cuda_reduce_kernel(T *input, T *output, int len) {
  //@@ Load a segment of the input vector into shared memory
  __shared__ T partialSum[2 * BLOCK_THREADS];
  unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_THREADS;
  if (start + t < len) {
    partialSum[t] = input[start + t];
  } else {
    partialSum[t] = 0;
  }
  if (start + BLOCK_THREADS + t < len) {
    partialSum[BLOCK_THREADS + t] = input[start + BLOCK_THREADS + t];
  } else {
    partialSum[BLOCK_THREADS + t] = 0;
  }
  //@@ Traverse the reduction tree
  for (unsigned int stride = BLOCK_THREADS; stride >= 1; stride >>= 1) {
    __syncthreads();
    if (t < stride) {
      partialSum[t] += partialSum[t + stride];
    }
  }
  //@@ Write the computed sum of the block to the output vector at the
  //@@ correct index
  if (t == 0) {
    output[blockIdx.x] = partialSum[0];
  }
}
