#pragma once

#include "flags.hpp"
#include "init_cuda.hpp"
#include "logger.hpp"

static void init(int argc, char **argv) {
  utils::logger::console = spdlog::stdout_logger_mt(argv[0]);
  init_flags(argc, argv);
  init_cuda();
}
