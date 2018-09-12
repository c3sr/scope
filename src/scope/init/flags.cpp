#include "clara/clara.hpp"

#include "scope/init/flags.hpp"
#include "scope/utils/utils.hpp"
#include "scope/init/init.hpp"



DEFINE_FLAG_vec_int32(cuda_device_ids, {}, "The cuda devices to use");
DEFINE_FLAG_bool(help, false, "Show help message.");
DEFINE_FLAG_int32(verbose, 1, "Verbose level.");
DEFINE_FLAG_bool(version, false, "Show version message.");

void register_flags() {
    RegisterOpt(clara::Help(FLAG(help)));
    RegisterOpt(clara::Opt(FLAG(verbose), "verbosity")["-v"]["--verbose"]("verbose mode"));
    RegisterOpt(clara::Opt(FLAG(version))["--version"]("print version info"));
    RegisterOpt(clara::Opt(FLAG(cuda_device_ids), "id")["-c"]["--cuda"]("add cuda device id"));
}
