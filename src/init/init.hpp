#pragma once

#include <cuda_runtime.h>

#include "logger.hpp"

extern bool has_cuda = false;
extern cudaDeviceProp cuda_device_prop;

void init(int argc, char **argv);
