#pragma once

#include "clara/clara.hpp"
#include "cuda.hpp"

void init_flags(int argc, char **argv);
void init();
void do_before_inits();

// return non-zero if program should exit with that code
typedef int (*InitFn)();

// a function that will run before the InitFns.
// It should register command line options and a version string
typedef void (*BeforeInitFn)();

void RegisterInit(InitFn fn);
void RegisterBeforeInit(BeforeInitFn fn);

// a string that will be returned by later calls to VersionStrings()
void RegisterVersionString(const std::string &s);

const std::vector<std::string>& VersionStrings();

#define SCOPE_REGISTER_INIT(x) static InitRegisterer _r_init_##x(x);

struct InitRegisterer {
  InitRegisterer(InitFn fn) {
    RegisterInit(fn);
  }
};

#define SCOPE_REGISTER_BEFORE_INIT(x) static BeforeInitRegisterer _r_before_init_##x(x);

struct BeforeInitRegisterer {
  BeforeInitRegisterer(BeforeInitFn fn) {
    RegisterBeforeInit(fn);
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
