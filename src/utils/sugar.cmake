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
include(sugar_include)

sugar_include(io)
sugar_include(tensor)

sugar_files(
    BENCHMARK_HEADERS
    commandlineflags.hpp
    hostname.hpp
    mpl.hpp
    numa.hpp
    timer.hpp
    cuda.hpp
    marker.hpp
    memory.hpp
    cudnn.hpp
    nvptx.hpp
    compat.hpp
    transwarp.h
    cublas.hpp
    defer.hpp
    benchmark.hpp
    error.hpp
    nccl.hpp
    utils.hpp
    nocopy.hpp
)

sugar_files(
    BENCHMARK_SOURCES
    commandlineflags.cpp
)

