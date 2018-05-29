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
    utils.hpp
    mpl.hpp
    cudnn.hpp
    cublas.hpp
    nocopy.hpp
    timer.hpp
    transwarp.h
    page.hpp
    numa.hpp
    hostname.hpp
    defer.hpp
    commandlineflags.hpp
    marker.hpp
    benchmark.hpp
    omp.hpp
    memory.hpp
    compat.hpp
    nvptx.hpp
    cuda.hpp
    nccl.hpp
)

sugar_files(
    BENCHMARK_SOURCES
    commandlineflags.cpp
)

