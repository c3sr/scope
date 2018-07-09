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
    cublas.hpp
    cuda.hpp
    cudnn.hpp
    flags.hpp
    init.hpp
    logger.hpp
    numa.hpp
)

sugar_files(
    BENCHMARK_SOURCES
    cublas.cpp
    cuda.cpp
    cudnn.cpp
    flags.cpp
    init.cpp
    logger.cpp
    numa.cpp
)

