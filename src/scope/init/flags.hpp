#pragma once


#include "scope/utils/commandlineflags.hpp"
#include "clara/clara.hpp"

DECLARE_FLAG_vec_int32(cuda_device_ids)
DECLARE_FLAG_bool(help)
DECLARE_FLAG_int32(verbose)
DECLARE_FLAG_bool(version)



void register_flags();
