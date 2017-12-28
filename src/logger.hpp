#pragma once

#include "spdlog/spdlog.h"

namespace utils {
namespace logger {
extern std::shared_ptr<spdlog::logger> console;
}
} // namespace utils

#define LOG(level, ...) utils::logger::console->level(__VA_ARGS__)

