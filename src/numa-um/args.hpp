

// #define TEST_ARGS() Ranges({{1<<10, 8<<10}, {0,1,2,3,4,5,6,7,8}, {0,1,2,3,4,5,6,7,8}})

// #define SMALL_ARGS() DenseRange(8, 31, 2)->ArgName("log2(N)")

// #define ALL_ARGS() DenseRange(1, 31, 1)->ArgName("log2(N)")

inline
static void ArgsCountNumaGpu(benchmark::internal::Benchmark* b) {

  int n;
  cudaError_t err = cudaGetDeviceCount(&n);
  if (PRINT_IF_ERROR(cudaGetDeviceCount(&n))) {
    exit(1);
  }

  for (auto numa_id : numa_nodes()) {
    for (int gpu_id = 0; gpu_id < n; ++gpu_id) {
      for (int j = 8; j <= 31; ++j) {
        b->Args({j, numa_id, gpu_id});
      }
    }
  }
}

inline
static void ArgsCountGpuGpuNoSelf(benchmark::internal::Benchmark* b) {

  int n;
  cudaError_t err = cudaGetDeviceCount(&n);
  if (PRINT_IF_ERROR(cudaGetDeviceCount(&n))) {
    exit(1);
  }

  for (int gpu0 = 0; gpu0 < n; ++gpu0) {
    for (int gpu1 = 0; gpu1 < n; ++gpu1) {
      if (gpu0 != gpu1) {
        for (int j = 8; j <= 31; ++j) {
          b->Args({j, gpu0, gpu1});
        }
      }
    }
  }
}