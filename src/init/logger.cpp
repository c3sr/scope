#include "config.hpp"

#include "logger.hpp"

#define XSTRINGIFY(s) STRINGIFY(s)
#define STRINGIFY(s) #s

std::shared_ptr<spdlog::logger> utils::logger::console = spdlog::stdout_logger_mt(XSTRINGIFY(PROJECT_NAME));
