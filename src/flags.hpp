#pragma once

#include <iostream>

#include "benchmark/benchmark.h"
#include "commandlineflags.hpp"
#include "config.hpp"

DECLARE_int32(cuda_device_id);

extern void init_flags(int argc, char **argv);
