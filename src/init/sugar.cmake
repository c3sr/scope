# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED SRC_INIT_SUGAR_CMAKE_)
  return()
else()
  set(SRC_INIT_SUGAR_CMAKE_ 1)
endif()

include(sugar_files)

sugar_files(
    BENCHMARK_HEADERS
    cudnn.hpp
    init.hpp
    cuda.hpp
    logger.hpp
    cublas.hpp
    flags.hpp
)

sugar_files(
    BENCHMARK_SOURCES
    cublas.cpp
    cudnn.cpp
    init.cpp
    logger.cpp
    cuda.cpp
    flags.cpp
)

