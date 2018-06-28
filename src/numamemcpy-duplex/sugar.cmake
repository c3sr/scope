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

sugar_files(
    BENCHMARK_HEADERS
    args.hpp
)

sugar_files(
    BENCHMARK_SOURCES
    host_to_gpu.cpp
)

