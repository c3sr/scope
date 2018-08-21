#include "scope/init/init.hpp"

#include "scope/init/cuda.hpp"
#include "scope/init/flags.hpp"
#include "scope/init/logger.hpp"

static struct { InitFn fn; } inits[10000];
static size_t ninits = 0;

void init(int argc, char** argv) {
  if (!bench::init::logger::console || bench::init::logger::console->name() != std::string(argv[0])) {
    bench::init::logger::console = spdlog::stdout_logger_mt(argv[0]);
  }
  init_flags(argc, argv);
  
  // keep levels somewhat consistent with Benchmark
  if (bench::flags::verbose == 0) {
    bench::init::logger::console->set_level(spdlog::level::off);
  } else if (bench::flags::verbose == 1) {
    bench::init::logger::console->set_level(spdlog::level::critical);
  } else if (bench::flags::verbose == 2) {
    bench::init::logger::console->set_level(spdlog::level::err);
  } else if (bench::flags::verbose == 3) {
    bench::init::logger::console->set_level(spdlog::level::debug);
  } else {
    bench::init::logger::console->set_level(spdlog::level::trace);
  }

  init_cuda();

  for (size_t i = 0; i < ninits; ++i) {
    LOG(debug, "Running registered initialization function...");
    int status = inits[i].fn(argc, argv);
    if (status) {
      exit(status);
    }
  }
}

void RegisterInit(InitFn fn) {
  if (ninits >= sizeof(inits) / sizeof(inits[0])) {
    LOG(critical, "ERROR: {}@{}: RegisterInit failed", __FILE__, __LINE__);
    exit(-1);
  }
  inits[ninits].fn = fn;
  ninits++;
}