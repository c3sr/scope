
#include "benchmark/benchmark.h"
#include "commandlineflags.hpp"

DEFINE_int32(cuda_device_id, 1, "The cuda device id to use.");
DEFINE_int32(verbose, 1, "Verbose level.");

static void parse(int* argc, char** argv) {
  using namespace utils;
  for (int i = 1; i < *argc; ++i) {
    if (ParseInt32Flag(argv[i], "cuda_device_id", &FLAGS_cuda_device_id) ||
        ParseInt32Flag(argv[i], "v", &FLAGS_verbose)) {
      for (int j = i; j != *argc - 1; ++j)
        argv[j] = argv[j + 1];

      --(*argc);
      --i;
    }
  }
}

void init_flags(int argc, char** argv) {
  parse(&argc, argv);

  benchmark::Initialize(&argc, argv);

  return;
}
