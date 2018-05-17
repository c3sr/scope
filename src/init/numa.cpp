
#include <tuple>

#include "optional/optional.hpp"

#include "init/flags.hpp"
#include "init/init.hpp"
#include "init/logger.hpp"

#include "init/numa.hpp"

bool has_numa = false;


bool init_numa() {

  int ret = numa_available();
  if (-1 == ret ) {
    LOG(critical, "NUMA not available");
    exit(1);
  } else {
      has_numa = true;
  }

  numa_set_strict(1);

  return false;
}
