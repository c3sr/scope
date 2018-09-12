#pragma once

#include "clara/clara.hpp"
#include "cuda.hpp"

void init_flags(int argc, char **argv);
void init();

// return non-zero if program should exit with that code
typedef int (*InitFn)();

void RegisterInit(InitFn fn);

#define SCOPE_REGISTER_INIT(x) static InitRegisterer _r_init_##x(x);

struct InitRegisterer {
  InitRegisterer(InitFn fn) {
    RegisterInit(fn);
  }
};


void RegisterOpt(clara::Opt opt);

struct OptRegisterer {

  template<typename ... Types>
  OptRegisterer(Types ... opts) {

    std::vector<clara::Opt> the_opts = {opts...};

    for (auto o : the_opts) {
      RegisterOpt(o);
    }
  }

};

#define SCOPE_REGISTER_OPTS(...) \
static OptRegisterer _reg (__VA_ARGS__)
