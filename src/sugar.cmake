# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED SRC_SUGAR_CMAKE_)
  return()
else()
  set(SRC_SUGAR_CMAKE_ 1)
endif()

include(sugar_files)
include(sugar_include)

<<<<<<< HEAD
sugar_include(um-coherence)
sugar_include(gemv)
sugar_include(numamemcpy-duplex)
sugar_include(numaum-latency)
sugar_include(numamemcpy)
sugar_include(launch)
sugar_include(utils)
sugar_include(example)
sugar_include(lock)
=======
sugar_include(atomic)
>>>>>>> 30886528961fead955a1713a60428ee8bb740ff0
sugar_include(axpy)
sugar_include(conv)
sugar_include(example)
sugar_include(gemm)
sugar_include(gemv)
sugar_include(init)
sugar_include(launch)
sugar_include(lock)
sugar_include(memcpy)
sugar_include(memcpy-duplex)
sugar_include(numa)
sugar_include(numamemcpy)
sugar_include(numaum-coherence)
sugar_include(numaum-latency)
sugar_include(numaum-prefetch)
sugar_include(um-coherence)
sugar_include(um-prefetch)
sugar_include(utils)
sugar_include(vectoradd)

sugar_files(
    BENCHMARK_HEADERS
    config.hpp
)

sugar_files(
    BENCHMARK_SOURCES
    main.cpp
)

