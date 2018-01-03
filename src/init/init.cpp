#include "init/init.hpp"

#include "utils/flags.hpp"
#include "init/cuda.hpp"
#include "init/logger.hpp"

bool has_cuda = false;
cudaDeviceProp cuda_device_prop;

void init(int argc, char **argv) {
  utils::logger::console = spdlog::stdout_logger_mt(argv[0]);
  init_flags(argc, argv);
  init_cuda();
}
