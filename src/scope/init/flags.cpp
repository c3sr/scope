
#include "scope/utils/utils.hpp"

DEFINE_vec_int32(cuda_device_ids, {}, "The cuda devices to use");
// DEFINE_bool(fast, false, "Whether to run only parts of the tests.");
DEFINE_int32(verbose, 1, "Verbose level.");
DEFINE_bool(help, false, "Show help message.");
DEFINE_bool(version, false, "Show version message.");

static void parse(int* argc, char** argv) {
  using namespace utils;
  for (int i = 1; i < *argc; ++i) {
    if (ParseVecInt32Flag(argv[i], "cuda_device_ids", &FLAG(cuda_device_ids)) || ParseBoolFlag(argv[i], "h", &FLAG(help)) ||
        ParseBoolFlag(argv[i], "help", &FLAG(help)) || ParseInt32Flag(argv[i], "v", &FLAG(verbose))) {
      for (int j = i; j != *argc - 1; ++j)
        argv[j] = argv[j + 1];

      --(*argc);
      --i;
    }   
  
  // don't consume version, so scopes may also see it and print a version string
  ParseBoolFlag(argv[i], "version", &FLAG(version));
  }

}

void init_flags(int argc, char** argv) {
  parse(&argc, argv);
  return;
}
