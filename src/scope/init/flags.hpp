#pragma once

#include "scope/utils/commandlineflags.hpp"

DECLARE_vec_int32(cuda_device_ids);
DECLARE_bool(help);
DECLARE_int32(verbose);
DECLARE_bool(version);

extern void init_flags(int argc, char **argv);
