# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED SRC_UTILS_SUGAR_CMAKE_)
  return()
else()
  set(SRC_UTILS_SUGAR_CMAKE_ 1)
endif()

include(sugar_files)

sugar_files(
    BENCHMARK_HEADERS
    mpl.hpp
    hostname.hpp
    cuda.hpp
    error.hpp
    nvptx.hpp
    timer.hpp
    cublas.hpp
    benchmark.hpp
    cudnn.hpp
    nocopy.hpp
    nccl.hpp
    memory.hpp
    compat.hpp
    utils.hpp
    defer.hpp
    marker.hpp
    commandlineflags.hpp
)

sugar_files(
    BENCHMARK_SOURCES
    commandlineflags.cpp
)

