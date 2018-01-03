#pragma once

#include "config.hpp"

#include <cuda_runtime.h>

extern bool has_cuda;
extern cudaDeviceProp cuda_device_prop;

void init(int argc, char **argv);
