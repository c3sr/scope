# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED SRC_SCOPE_UTILS_SUGAR_CMAKE_)
  return()
else()
  set(SRC_SCOPE_UTILS_SUGAR_CMAKE_ 1)
endif()

include(sugar_files)
include(sugar_include)

sugar_include(io)
sugar_include(tensor)

sugar_files(
    BENCHMARK_HEADERS
    benchmark.hpp
    commandlineflags.hpp
    compat.hpp
    cublas.hpp
    cuda.hpp
    defer.hpp
    error.hpp
    hostname.hpp
    marker.hpp
    mpl.hpp
    nocopy.hpp
    nvptx.hpp
    page.hpp
    timer.hpp
    transwarp.h
    utils.hpp
)

sugar_files(
    BENCHMARK_SOURCES
    commandlineflags.cpp
)

