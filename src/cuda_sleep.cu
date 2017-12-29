#include <benchmark/benchmark.h>

#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <cuda_runtime.h>

#include "fmt/format.h"

#include "init.hpp"
#include "utils.hpp"
#include "utils_cuda.hpp"

using clock_value_t = long long;

// This is a kernel that does no real work but runs at least for a specified number of clocks
__device__ void sleep(clock_value_t sleep_cycles) {
  clock_value_t start = clock64();
  clock_value_t cycles_elapsed;

  // The code below should work like
  // this (thanks to modular arithmetics):
  //
  // clock_offset = (clock_t) (end_clock > start_clock ?
  //                           end_clock - start_clock :
  //                           end_clock + (0xffffffffu - start_clock));
  //
  // Indeed, let m = 2^32 then
  // end - start = end + m - start (mod m).
  do {
    cycles_elapsed = clock64() - start;
  } while (cycles_elapsed < sleep_cycles);
}

static int64_t get_cycles(float seconds) {
  // Get device frequency in KHz
  int64_t Hz;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  Hz = int64_t(prop.clockRate) * 1000;

  // Calculate number of cycles to wait
  int64_t num_cycles;
  num_cycles = (int64_t)(seconds * Hz);

  return num_cycles;
}