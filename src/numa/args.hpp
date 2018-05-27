inline
static void ArgsCountThreadsNumaNuma(benchmark::internal::Benchmark* b) {

  for (int j = 12; j <= 30; ++j) { // log2(bytes)
    for (int k = 1; k <= 8; k *= 2) { // threads
      for (auto src_numa : numa_nodes()) {
        for (auto dst_numa : numa_nodes()) {
          if (src_numa != dst_numa) {
            b->Args({j, k, src_numa, dst_numa});
          }
        }
      }
    }
  }
}

inline
static void ArgsCountNumaNuma(benchmark::internal::Benchmark* b) {

  for (int j = 12; j <= 30; ++j) { // log2(bytes)
      for (auto src_numa : numa_nodes()) {
        for (auto dst_numa : numa_nodes()) {
            b->Args({j, src_numa, dst_numa});
        }
      }
    }
}