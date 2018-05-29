#pragma once

#include <omp.h>

static void omp_numa_bind_node(const int numa_id) {
  numa_bind_node(numa_id);
#pragma omp parallel
  { numa_bind_node(numa_id); }
}
