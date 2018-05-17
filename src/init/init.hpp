#pragma once

#include "config.hpp"

#include <cuda_runtime.h>

#include "init/cublas.hpp"
#include "init/cuda.hpp"
#include "init/cudnn.hpp"
#include "init/numa.hpp"

void init(int argc, char **argv);
