# Benchmarks and Descriptions

The following micro-benchmarks are currently available

|Benchmarks|Description|Argument Format|
|-|-|-|
| CBLAS_DAXPY | CBLAS DAXPY operation | `<vec size>/<increment x>/<increment y>` |
| CUBLAS_CGEMM | `cublas_gemm`, single-precision complex | `m/n/k` |
| CUBLAS_DGEMM | `cublas_gemm`, double-precision         | `m/n/k` |
| CUBLAS_HGEMM | `cublas_gemm`, half-precision           | `m/n/k` |
| CUBLAS_SGEMM | `cublas_gemm`, single-precision         | `m/n/k` |
| CUBLAS_ZGEMM | `cublas_gemm`, double-precision complex | `m/n/k` |
| CUDA_LAUNCH_`[kernel]`_<`[type]`, `launch count`, `iter count`, `block size`> | kernel launch time | number of threads |
| CUDA_SGEMM_BASIC                | Custom CUDA basic matrix multiplication. | `m/n/k` |
| CUDA_SGEMM_TILED<`[tile size]`> | Custom CUDA shared-memory tiled matrix multiplication. | `m/n/k` |
| CUDA_VECTOR_ADD<`[dtype]`,`[coarsening]`,`[block size]`> | Custom CUDA vector addition with coarsening. | vector elements |
| CUDNN_CONV_`[dtype]`<`[algorithm]`> | CUDNN convolution | `width/height/channels/n/k/filter width/filter height/width padding/height padding/width stride/height stride` | 
| EXAMPLE1 | An example benchmark. | -- |
| EXAMPLE2 | An example benchmark. | -- |
| LOCK_`[method]` | lock acquire and release time | number of acquire/release attempts |
