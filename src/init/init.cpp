#include "init/init.hpp"

#include "init/cublas.hpp"
#include "init/cuda.hpp"
#include "init/cudnn.hpp"
#include "init/flags.hpp"
#include "init/logger.hpp"

static struct { InitFn fn; const char *name, *type; } inits[10000];
static size_t ninits = 0;

void init(int argc, char **argv) {
  bench::init::logger::console = spdlog::stdout_logger_mt(argv[0]);
  init_flags(argc, argv);
  init_cuda();
  init_cublas();
  init_cudnn();

  for (size_t i = 0; i < ninits; ++i) {
    inits[i].fn(argc, argv);
  }
}

void RegisterInit(InitFn fn, const char* name, const char* type) {
	if(ninits >= sizeof(inits)/sizeof(inits[0])) {
		printf("%s %s, line %d: RegisterInit failed\n", name, __FILE__, __LINE__);
		exit(-1);
	}
	inits[ninits].fn = fn;
	inits[ninits].name = name;
	inits[ninits].type = type;
	ninits++;
}