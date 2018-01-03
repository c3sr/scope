#pragma once

#include <cuda_runtime.h>

#include "init/logger.hpp"
#include "init/cublas.hpp"
#include "init/cuda.hpp"

extern bool has_cuda;
extern cudaDeviceProp cuda_device_prop;

void init(int argc, char **argv);
