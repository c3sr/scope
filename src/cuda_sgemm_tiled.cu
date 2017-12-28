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
  //@@ Insert code to implement matrix multiplication here
  __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
  __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y,
      Row = by * TILE_WIDTH + ty, Col = bx * TILE_WIDTH + tx;
  float Pvalue = 0;

  for (int m = 0; m < (numAColumns - 1) / TILE_WIDTH + 1; ++m) {
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

    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; ++k)
      Pvalue += ds_M[ty][k] * ds_N[k][tx];
    __syncthreads();
  }
  if (Row < numCRows && Col < numCColumns)
    C[Row * numCColumns + Col] = Pvalue;
}
