

#include <benchmark/benchmark.h>

#include "config.hpp"
#include "flags.hpp"
#include "init.hpp"
#include "utils.hpp"

const char *bench_git_refspec();
const char *bench_git_hash();
const char *bench_git_tag();

std::string bench_get_build_version() {
  return fmt::format("{} {} {}", bench_git_refspec(), bench_git_tag(), bench_git_hash());
}

int main(int argc, char **argv) {
  init(argc, argv);
  // if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
}
