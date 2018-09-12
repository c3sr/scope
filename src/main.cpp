
#include "config.hpp"

#include <benchmark/benchmark.h>

#include "scope/init/flags.hpp"
#include "scope/init/init.hpp"
#include "scope/utils/utils.hpp"
#include "scope/utils/version.hpp"

static const auto benchmark_help = R"(
The following options are also recognized according to Google Benchmark:
      [--benchmark_list_tests={true|false}]
      [--benchmark_filter=<regex>]
      [--benchmark_min_time=<min_time>]
      [--benchmark_repetitions=<num_repetitions>]
      [--benchmark_report_aggregates_only={true|false}
      [--benchmark_format=<console|json|csv>]
      [--benchmark_out=<filename>]
      [--benchmark_out_format=<json|console|csv>]
      [--benchmark_color={auto|true|false}]
      [--benchmark_counters_tabular={true|false}]
)";

std::vector<char*> get_benchmark_argv(int argc, char *const *argv) {
  std::vector<char*> ret;
  ret.push_back(argv[0]);
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg.find("--benchmark_") == 0) {
      ret.push_back(argv[i]);
    }
  }
  return ret;
}

std::vector<char*> get_scope_argv(int argc, char *const *argv) {
  std::vector<char*> ret;
  ret.push_back(argv[0]);
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg.find("--benchmark_") == std::string::npos) {
      ret.push_back(argv[i]);
    }
  }
  return ret;
}

int main(int argc, char **argv) {

  if (!bench::init::logger::console || bench::init::logger::console->name() != std::string(argv[0])) {
    bench::init::logger::console = spdlog::stdout_logger_mt(argv[0]);
  }

  // run all the registered before_inits
  do_before_inits();

  // Separate out the arguments
  std::vector<char*> benchmark_argv = get_benchmark_argv(argc, argv);
  std::vector<char*> scope_argv = get_scope_argv(argc, argv);

  // register scope flags and parse all flags
  init_flags(scope_argv.size(), scope_argv.data());

  // keep levels somewhat consistent with Benchmark
  if (FLAG(verbose) == 0) {
    bench::init::logger::console->set_level(spdlog::level::off);
  } else if (FLAG(verbose) == 1) {
    bench::init::logger::console->set_level(spdlog::level::critical);
  } else if (FLAG(verbose) == 2) {
    bench::init::logger::console->set_level(spdlog::level::err);
  } else if (FLAG(verbose) == 3) {
    bench::init::logger::console->set_level(spdlog::level::debug);
  } else {
    bench::init::logger::console->set_level(spdlog::level::trace);
  }

   if (FLAG(help)) {
     std::cout << benchmark_help << "\n";
     return 0;
  }

  if (FLAG(version)) {
    // print scope version
    std::cout << version("Scope", SCOPE_VERSION, SCOPE_GIT_REFSPEC, SCOPE_GIT_HASH, SCOPE_GIT_LOCAL_CHANGES) << "\n";
    for (auto version_str : VersionStrings()) {
      std::cout << version_str << "\n";
    }
    return 0;
  }

  // run main init functions
  init();

  //if (::benchmark::ReportUnrecognizedArguments(benchmark_argv.size(), benchmark_argv.data())) return 1;

  int i = benchmark_argv.size();
  benchmark::Initialize(&i, benchmark_argv.data());


  // auto options = benchmark::internal::GetOutputOptions();

  // benchmark::ConsoleReporter CR(options);
  // benchmark::JSONReporter JR;
  // benchmark::CSVReporter CSVR;

  benchmark::RunSpecifiedBenchmarks();
}
