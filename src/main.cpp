
#include "config.hpp"

#include <benchmark/benchmark.h>

#include "scope/init/flags.hpp"
#include "scope/init/init.hpp"
#include "scope/utils/utils.hpp"
#include "scope/utils/version.hpp"

static const auto help = R"(
scope [--benchmark_list_tests={true|false}]
      [--benchmark_filter=<regex>]
      [--benchmark_min_time=<min_time>]
      [--benchmark_repetitions=<num_repetitions>]
      [--benchmark_format=<tabular|json|csv>]
      [--color_print={true|false}]
      [--v=<verbosity>]
      [ --version]
)";

int main(int argc, char **argv) {
  init(argc, argv);
  // if (FLAG(help)) {
  //   std::cout << help << "\n";
  //   return 0;
  // }

  if (FLAG(version)) {
    std::cout << version("Scope", SCOPE_VERSION, SCOPE_GIT_REFSPEC, SCOPE_GIT_HASH, SCOPE_GIT_LOCAL_CHANGES) << "\n";
    return 0;
  }



  benchmark::Initialize(&argc, argv);

  // if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;

  // auto options = benchmark::internal::GetOutputOptions();

  // benchmark::ConsoleReporter CR(options);
  // benchmark::JSONReporter JR;
  // benchmark::CSVReporter CSVR;

  benchmark::RunSpecifiedBenchmarks();
}
