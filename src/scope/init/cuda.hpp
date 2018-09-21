#pragma once

#include <vector>

#include <cuda_runtime.h>

#include "scope/utils/utils.hpp"

extern bool has_cuda;
extern cudaDeviceProp cuda_device_prop;

bool init_cuda();
int num_gpus();

// Valid and unique CUDA device ids passed to scope
// passing -c 0 -c 0 will yield [0]
const std::vector<int> &unique_cuda_device_ids();
