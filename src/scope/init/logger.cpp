#include "logger.hpp"

std::shared_ptr<spdlog::logger> bench::init::logger::console = spdlog::stdout_logger_mt("scope");
