# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED SRC_NUMAMEMCPY_SUGAR_CMAKE_)
  return()
else()
  set(SRC_NUMAMEMCPY_SUGAR_CMAKE_ 1)
endif()

include(sugar_files)

sugar_files(
    BENCHMARK_HEADERS
    args.hpp
)

sugar_files(
    BENCHMARK_SOURCES
    pinned_to_gpu.cpp
    gpu_to_gpu_nopeer.cpp
    gpu_to_host.cpp
    gpu_to_pinned.cpp
    wc_to_gpu.cpp
    gpu_to_wc.cpp
    host_to_gpu.cpp
    host_to_pinned.cpp
)

