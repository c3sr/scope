# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED INIT_SUGAR_CMAKE_)
  return()
else()
  set(INIT_SUGAR_CMAKE_ 1)
endif()

include(sugar_files)

sugar_files(
    BENCHMARK_HEADERS
    init.hpp
    logger.hpp
    cublas.hpp
    flags.hpp
    numa.hpp
    cudnn.hpp
    cuda.hpp
)

sugar_files(
    BENCHMARK_SOURCES
    init.cpp
    cudnn.cpp
    logger.cpp
    cublas.cpp
    numa.cpp
    flags.cpp
    cuda.cpp
)

