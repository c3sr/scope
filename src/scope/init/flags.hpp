#pragma once

#include <iostream>

#include "config.hpp"

#include "scope/utils/commandlineflags.hpp"

DECLARE_int32(cuda_device_id);
DECLARE_bool(help);
DECLARE_int32(verbose);
DECLARE_bool(version);

extern void init_flags(int argc, char **argv);
