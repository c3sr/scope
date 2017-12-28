#include "config.hpp"

#include "logger.hpp"

std::shared_ptr<spdlog::logger> utils::logger::console =
    spdlog::stdout_logger_mt(PROJECT_NAME);
