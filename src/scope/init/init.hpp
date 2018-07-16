#pragma once

#include "config.hpp"

#include <cuda_runtime.h>

#include "cublas.hpp"
#include "cuda.hpp"
#include "cudnn.hpp"

void init(int argc, char **argv);

typedef void (*InitFn)(int argc, char **argv);

void RegisterInit(InitFn fn, const char *name, const char *type);

#define INIT(x) static InitRegisterer _r_init_##x(x, #x ".", "init");

struct InitRegisterer {
  InitRegisterer(InitFn fn, const char *name, const char *type) {
    RegisterInit(fn, name, type);
  }
};