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

sugar_include(tensor)
sugar_include(io)

sugar_files(
    BENCHMARK_HEADERS
    error.hpp
    commandlineflags.hpp
    nvptx.hpp
    hostname.hpp
    timer.hpp
    cudnn.hpp
    mpl.hpp
    transwarp.h
    defer.hpp
    memory.hpp
    utils.hpp
    benchmark.hpp
    compat.hpp
    nocopy.hpp
    cuda.hpp
    cublas.hpp
    marker.hpp
    nccl.hpp
)

sugar_files(
    BENCHMARK_SOURCES
    commandlineflags.cpp
)

