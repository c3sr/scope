#include "scope/init/init.hpp"

#include "scope/init/cublas.hpp"
#include "scope/init/cuda.hpp"
#include "scope/init/flags.hpp"
#include "scope/init/logger.hpp"

static struct { InitFn fn; } inits[10000];
static size_t ninits = 0;

void init(int argc, char** argv) {
  bench::init::logger::console = spdlog::stdout_logger_mt(argv[0]);
  init_flags(argc, argv);
// enum level_enum
// {
//     trace = 0,
//     debug = 1,
//     info = 2,
//     warn = 3,
//     err = 4,
//     critical = 5,
//     off = 6
// };
  
  // keep levels somewhat consistent with Benchmark
  if (bench::flags::verbose == 0) {
    bench::init::logger::console->set_level(spdlog::level::off);
  } else if (bench::flags::verbose == 1) {
    bench::init::logger::console->set_level(spdlog::level::err);
  } else if (bench::flags::verbose == 2) {
    bench::init::logger::console->set_level(spdlog::level::err);
  } else if (bench::flags::verbose == 3) {
    bench::init::logger::console->set_level(spdlog::level::debug);
  } else {
    bench::init::logger::console->set_level(spdlog::level::trace);
  }

  init_cuda();
  init_cublas();

  for (size_t i = 0; i < ninits; ++i) {
    LOG(debug, "Running registered initialization function...");
    inits[i].fn(argc, argv);
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