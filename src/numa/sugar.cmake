# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED NUMA_SUGAR_CMAKE_)
  return()
else()
  set(NUMA_SUGAR_CMAKE_ 1)
endif()

include(sugar_files)

sugar_files(
    BENCHMARK_HEADERS
    args.hpp
    ops.hpp
)

sugar_files(
    BENCHMARK_SOURCES
    rd.cpp
    wr.cpp
    ops.cpp
)

