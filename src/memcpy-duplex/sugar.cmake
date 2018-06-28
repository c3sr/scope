# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

<<<<<<< HEAD
if(DEFINED MEMCPY_DUPLEX_SUGAR_CMAKE_)
  return()
else()
  set(MEMCPY_DUPLEX_SUGAR_CMAKE_ 1)
=======
if(DEFINED SRC_MEMCPY_DUPLEX_SUGAR_CMAKE_)
  return()
else()
  set(SRC_MEMCPY_DUPLEX_SUGAR_CMAKE_ 1)
>>>>>>> 30886528961fead955a1713a60428ee8bb740ff0
endif()

include(sugar_files)

sugar_files(
    BENCHMARK_HEADERS
    args.hpp
)

sugar_files(
    BENCHMARK_SOURCES
    gpu_gpu.cpp
)

