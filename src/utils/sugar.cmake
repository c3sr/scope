# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED UTILS_SUGAR_CMAKE_)
  return()
else()
  set(UTILS_SUGAR_CMAKE_ 1)
endif()

include(sugar_files)
include(sugar_include)

sugar_include(io)
sugar_include(tensor)

sugar_files(
    BENCHMARK_HEADERS
    error.hpp
    mpl.hpp
    marker.hpp
    memory.hpp
    omp.hpp
    benchmark.hpp
    transwarp.h
    nocopy.hpp
    cublas.hpp
    nvptx.hpp
    nccl.hpp
    timer.hpp
    utils.hpp
    commandlineflags.hpp
    page.hpp
    hostname.hpp
    compat.hpp
    numa.hpp
    defer.hpp
    cudnn.hpp
    cuda.hpp
)

sugar_files(
    BENCHMARK_SOURCES
    commandlineflags.cpp
)

