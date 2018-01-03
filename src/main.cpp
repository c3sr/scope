

#include <benchmark/benchmark.h>

#include "config.hpp"
#include "flags.hpp"
#include "init.hpp"
#include "utils.hpp"

const char *bench_git_refspec();
const char *bench_git_hash();
const char *bench_git_tag();


static const auto help = R"(
benchmark [--benchmark_list_tests={true|false}]
          [--benchmark_filter=<regex>]
          [--benchmark_min_time=<min_time>]
          [--benchmark_repetitions=<num_repetitions>]
          [--benchmark_format=<tabular|json|csv>]
          [--color_print={true|false}]
          [--v=<verbosity>]
)";

std::string bench_get_build_version() {
  return fmt::format("{} {} {}", bench_git_refspec(), bench_git_tag(), bench_git_hash());
}

int main(int argc, char **argv) {
  init(argc, argv);
  // if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;

  auto options = benchmark::internal::GetOutputOptions();
  
  benchmark::ConsoleReporter CR(options);
  benchmark::JSONReporter JR;
  benchmark::CSVReporter CSVR;

  benchmark::RunSpecifiedBenchmarks(CR, JR);
}
