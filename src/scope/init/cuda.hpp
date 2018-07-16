#pragma once

#include <cuda_runtime.h>

#include "scope/utils/utils.hpp"

extern bool has_cuda;
extern cudaDeviceProp cuda_device_prop;

bool init_cuda();
