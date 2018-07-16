# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED SUGAR_CMAKE_)
  return()
else()
  set(SUGAR_CMAKE_ 1)
endif()

include(sugar_files)
include(sugar_include)

sugar_include(atomic)
sugar_include(axpy)
sugar_include(conv)
sugar_include(example)
sugar_include(gemm)
sugar_include(gemv)
sugar_include(launch)
sugar_include(lock)
sugar_include(scope)
sugar_include(vectoradd)

sugar_files(
    BENCHMARK_HEADERS
    config.hpp
)

sugar_files(
    BENCHMARK_SOURCES
    main.cpp
)

