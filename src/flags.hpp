#pragma once

#include <cxxopts.hpp>

#include "config.hpp"

static int cuda_device_id = 0;

static void init_flags(int argc, char **argv) {
  cxxopts::Options options(argv[0], "microbenchmark suite");

  options.add_options()("d,debug", "Enable debugging")(
      "cuda_device", "CUDA device to use",
      cxxopts::value<int>()->default_value("0"))("f,file", "File name",
                                                 cxxopts::value<std::string>());

  auto result = options.parse(argc, argv);

  cuda_device_id = result["cuda_device"].as<int>();
}
