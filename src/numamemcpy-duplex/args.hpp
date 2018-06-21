#include <cstdint>
#include <cstdlib>
//is this required for duplex
inline static size_t popcount(uint64_t u) {
  return __builtin_popcount(u);
}

inline static bool is_set(uint64_t bits, size_t i) {
  return (uint64_t(1) << i) & bits;
}

//need some argscount from host to gpu, maybe something like this
inline static void ArgsCountNumaGpu(benchmark::internal::Benchmark* b) {

  int n;
  cudaError_t err = cudaGetDeviceCount(&n);
  if (PRINT_IF_ERROR(cudaGetDeviceCount(&n))) {
    exit(1);
  }

  for (auto numa : numa_nodes()) {
    for (int gpu1 = 0; gpu1 < n; ++gpu1) {
      for (int j = 8; j <= 33; ++j) {
        b->Args({j, numa, gpu1});
      }
    }
  }
}
