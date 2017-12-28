#pragma once

#include <cxxopts.hpp>
#include <iostream>

#include "config.hpp"

static int cuda_device_id = 0;

static void init_flags(int argc, char **argv) {
  try {
    cxxopts::Options options(argv[0], "microbenchmark suite");

    options.positional_help("[optional args]").show_positional_help();
    options.add_options()("d,debug", "Enable debugging")(
        "cuda_device", "CUDA device to use",
        cxxopts::value<int>()->default_value("0"))(
        "f,file", "File name", cxxopts::value<std::string>());

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
      std::cout << options.help({}) << std::endl;
      exit(0);
    }

    cuda_device_id = result["cuda_device"].as<int>();
  } catch (const cxxopts::OptionException &e) {
    // std::cout << "error parsing options: " << e.what() << std::endl;
  }

  return;
}
