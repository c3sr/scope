
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
