#pragma once

#include "cuda.hpp"

void init(int argc, char **argv);

typedef int (*InitFn)(int argc, char *const * argv);

void RegisterInit(InitFn fn);

#define SCOPE_INIT(x) static InitRegisterer _r_init_##x(x);

struct InitRegisterer {
  InitRegisterer(InitFn fn) {
    RegisterInit(fn);
  }
};
