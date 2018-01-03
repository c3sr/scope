#include "init.hpp"

#include "flags.hpp"
#include "init_cuda.hpp"
#include "logger.hpp"

bool has_cuda = false;
cudaDeviceProp cuda_device_prop;

void init(int argc, char **argv) {
  utils::logger::console = spdlog::stdout_logger_mt(argv[0]);
  init_flags(argc, argv);
  init_cuda();
}
