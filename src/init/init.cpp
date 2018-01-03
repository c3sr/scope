#include "init/init.hpp"

#include "init/cuda.hpp"
#include "init/flags.hpp"
#include "init/logger.hpp"

void init(int argc, char **argv) {
  bench::init::logger::console = spdlog::stdout_logger_mt(argv[0]);
  init_flags(argc, argv);
  init_cuda();
}
