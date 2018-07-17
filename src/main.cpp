
#include "config.hpp"

#include <benchmark/benchmark.h>

#include "scope/init/flags.hpp"
#include "scope/init/init.hpp"
#include "scope/utils/utils.hpp"

static const auto help = R"(
benchmark [--benchmark_list_tests={true|false}]
          [--benchmark_filter=<regex>]
          [--benchmark_min_time=<min_time>]
          [--benchmark_repetitions=<num_repetitions>]
          [--benchmark_format=<tabular|json|csv>]
          [--color_print={true|false}]
          [--v=<verbosity>]
)";

// const char *bench_git_refspec();
// const char *bench_git_hash();
// const char *bench_git_tag();

// std::string bench_get_build_version() {
//   return fmt::format("{} {} {}", bench_git_refspec(), bench_git_tag(), bench_git_hash());
// }

int main(int argc, char **argv) {
  init(argc, argv);

  // if (FLAG(help)) {
  //   std::cout << help << "\n";
  //   return 0;
  // }

  benchmark::Initialize(&argc, argv);

  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;

  // auto options = benchmark::internal::GetOutputOptions();

  // benchmark::ConsoleReporter CR(options);
  // benchmark::JSONReporter JR;
  // benchmark::CSVReporter CSVR;

  benchmark::RunSpecifiedBenchmarks();
}
