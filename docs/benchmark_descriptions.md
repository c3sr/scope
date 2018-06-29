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
| CUDA_Memcpy_GPUToGPU    | `cudaMemcpy` GPU allocation to pageable host allocation, peer access enabled | `log2 bytes` |
| CUDA_Memcpy_GPUToHost   | `cudaMemcpy` GPU allocation to pageable host allocation, no NUMA control     | `log2 bytes` |
| CUDA_Memcpy_GPUToPinned | `cudaMemcpy` GPU allocation to pinned host allocation, no NUMA control       | `log2 bytes` |
| CUDA_Memcpy_GPUToPinned | `cudaMemcpy` GPU allocation to pinned host allocation, no NUMA control       | `log2 bytes` |
| CUDA_Memcpy_HostToGPU   | `cudaMemcpy` pageable host allocation to GPU allocation                      | `log2 bytes` |
| CUDA_Memcpy_PinnedToGPU | `cudaMemcpy` pinned host allocation to GPU allocation, no NUMA control       | `log2 bytes` |
| CUDA_SGEMM_BASIC                | Custom CUDA basic matrix multiplication. | `m/n/k` |
| CUDA_SGEMM_TILED<`[tile size]`> | Custom CUDA shared-memory tiled matrix multiplication. | `m/n/k` |
| CUDA_VECTOR_ADD<`[dtype]`,`[coarsening]`,`[block size]`> | Custom CUDA vector addition with coarsening. | vector elements |
| CUDNN_CONV_`[dtype]`<`[algorithm]`> | CUDNN convolution | `width/height/channels/n/k/filter width/filter height/width padding/height padding/width stride/height stride` | 
| EXAMPLE1 | An example benchmark. | -- |
| EXAMPLE2 | An example benchmark. | -- |
| NUMAUM_Coherence_HostToGPU        | Host-to-GPU CUDA unified memory coherence transfer with NUMA pinning. | `log2 bytes/NUMA/GPU` |
| NUMAUM_Coherence_GPUToHost        | GPU-to-Host CUDA unified memory coherence transfer with NUMA pinning. | `log2 bytes/NUMA/GPU` |
| NUMAUM_Coherence_GPUToHostThreads | GPU-to-Host CUDA unified memory coherence transfer with NUMA pinning, multiple host threads. | `threads/log2 bytes/NUMA/GPU` |
| NUMAUM_Coherence_GPUThreads | Effect of GPU parallelism on h2d coherence transfer | `warps/NUMA/GPU`
| NUMAUM_Prefetch_GPUToHost | GPU-to-Host CUDA unified memory prefetch transfer with NUMA pinning. | `log2 bytes/NUMA/GPU` |
| NUMAUM_Prefetch_HostToGPU | Host-to-GPU CUDA unified memory prefetch transfer with NUMA pinning. | `log2 bytes/NUMA/GPU` |
| NUMAUM_Latency_HostToGPU | GPU-to-Host CUDA unified memory page fault measurement with NUMA pinning. | `stride count/NUMA/GPU` |
| NUMAUM_Latency_GPUToHost | Host-to-GPU CUDA unified memory page fault measurement with NUMA pinning. | `stride count/NUMA/GPU` |
| NUMA_Memcpy_GPUToGPU     | `cudaMemcpy` GPU allocation to GPU allocation with peer access disabled and NUMA pinning. | `<log2 bytes>/NUMA/<src GPU>/<dst GPU>`
| NUMA_Memcpy_GPUToHost    | `cudaMemcpy` GPU allocation to pageable Host allocation with NUMA pinning.                | `<log2 bytes>/<NUMA>/<GPU>`
| NUMA_Memcpy_GPUToPinned  | `cudaMemcpy` GPU allocation to pinned Host allocation with NUMA pinning.                  | `<log2 bytes>/<NUMA>/<GPU>`
| NUMA_Memcpy_GPUToWC      | `cudaMemcpy` GPU allocation to write-combining host allocation with NUMA pinning.         | `<log2 bytes>/<NUMA>/<GPU>`
| NUMA_Memcpy_HostToGPU    | `cudaMemcpy` pageable host allocation to GPu allocation with NUMA pinning.                | `<log2 bytes>/<NUMA>/<GPU>`
| NUMA_Memcpy_HostToPinned | `cudaMemcpy` Host pageable allocation to host pinned allocation with NUMA pinning.        | `<log2 bytes>/<src NUMA>/<dst NUMA>`
| NUMA_Memcpy_PinnedToGPU  | `cudaMemcpy` Pinned host allocation to GPU allocation NUMA pinning.                       | `<log2 bytes>/<NUMA>/<GPU>`
| NUMA_Memcpy_WCToGPU      | `cudaMemcpy` write-combined host allocation to GPU allocation with NUMA pinning.          | `<log2 bytes>/<NUMA>/<GPU>`
| NUMA_RD | OpenMP read from NUMA node | `<threads>/<log2 bytes><src NUMA><dst NUMA>` |
| NUMA_WR | OpenMP write to NUMA node  | `<threads>/<log2 bytes><src NUMA><dst NUMA>`|
| UM_Coherence_GPUToGPU | GPU-to-GPU CUDA unified memory coherence transfer. | `log2 bytes/src GPU/dst GPU` |
| UM_Latency_GPUToGPU | GPU-to-GPU CUDA unified memory page fault latency | `traversal count/src GPU/dst GPU` |
| UM_Prefetch_GPUToGPU  | GPU-to-GPU CUDA unified memory prefetch transfer.  | `log2 bytes/src GPU/dst GPU` |
| LOCK_`[method]` | lock acquire and release time | number of acquire/release attempts |
