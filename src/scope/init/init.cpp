#include "scope/init/init.hpp"
#include "scope/init/cuda.hpp"
#include "scope/init/flags.hpp"
#include "scope/init/logger.hpp"

static struct { InitFn fn; } inits[10000];
static size_t ninits = 0;

static BeforeInitFn before_inits[10000];
static size_t n_before_inits = 0;

static clara::Parser cli;
static std::vector<std::string> version_strings;

const std::vector<std::string>& VersionStrings() {
  return version_strings;
}

void RegisterOpt(clara::Opt opt) {
  cli = cli | opt;
}


void do_before_inits() {
  for (size_t i = 0; i < n_before_inits; ++i) {
    before_inits[i]();
  }
}

void init_flags(int argc, char **argv) {

  // register scope flags first
  register_flags();

  // remove all --benchmark flags so clara doesn't barf on them, and restore them after clara parsing
  std::vector<std::string> benchmark_flags;
  for (int i = 1; i < argc; ++i) {
    std::string opt(argv[i]);
    if (opt.find("--benchmark_") == 0) {
      benchmark_flags.push_back(opt);
      // shift remaining opts down by one
      for (int j = i; j < argc; ++j) {
        argv[j] = argv[j+1];
      }
      argc--;
    }
  }

  // parse remaining flags
  auto result = cli.parse(clara::Args(argc, argv));
  if (!result) {
    LOG(critical, result.errorMessage());
    exit(-1);
  }

  // restore --benchmark flags
  for (size_t i = 0; i < benchmark_flags.size(); ++i) {
    benchmark_flags[i].copy(argv[argc-1+i], benchmark_flags[i].size());
  }
  argc += benchmark_flags.size();

  if (FLAG(help)) {
    std::cout << cli << "\n";
  }

}

void init() {

  init_cuda();

  for (size_t i = 0; i < ninits; ++i) {
    LOG(debug, "Running registered initialization function...");
    int status = inits[i].fn();
    if (status) {
      exit(status);
    }
  }
}



void RegisterInit(InitFn fn) {
  if (ninits >= sizeof(inits) / sizeof(inits[0])) {
    LOG(critical, "ERROR: {}@{}: RegisterInit failed, too many inits", __FILE__, __LINE__);
    exit(-1);
  }
  inits[ninits].fn = fn;
  ninits++;
}

void RegisterBeforeInit(BeforeInitFn fn) {
  if (n_before_inits >= sizeof(before_inits) / sizeof(before_inits[0])) {
    LOG(critical, "ERROR: {}@{}: RegisterBeforeInit failed, too many functions", __FILE__, __LINE__);
    exit(-1);
  }
  before_inits[n_before_inits] = fn;
  n_before_inits++;
}

void RegisterVersionString(const std::string &s) {
  version_strings.push_back(s);
}