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
    numa.hpp
    cuda.hpp
    page.hpp
    timer.hpp
    compat.hpp
    memory.hpp
    hostname.hpp
    omp.hpp
    marker.hpp
    transwarp.h
    defer.hpp
    nocopy.hpp
    nccl.hpp
    utils.hpp
    commandlineflags.hpp
    nvptx.hpp
    benchmark.hpp
    mpl.hpp
    error.hpp
    cudnn.hpp
    cublas.hpp
)

sugar_files(
    BENCHMARK_SOURCES
    omp.cpp
    commandlineflags.cpp
)

