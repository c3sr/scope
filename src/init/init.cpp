#include "init/init.hpp"

#include "init/cublas.hpp"
#include "init/cuda.hpp"
#include "init/cudnn.hpp"
#include "init/flags.hpp"
#include "init/logger.hpp"
#include "init/numa.hpp"

void init(int argc, char **argv) {
  bench::init::logger::console = spdlog::stdout_logger_mt(argv[0]);
  init_flags(argc, argv);
  init_cuda();
  init_cublas();
  init_cudnn();
  init_numa();
}
