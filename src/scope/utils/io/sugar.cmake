# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED BENCHMARK_SRC_SCOPE_UTILS_IO_SUGAR_CMAKE_)
  return()
else()
  set(BENCHMARK_SRC_SCOPE_UTILS_IO_SUGAR_CMAKE_ 1)
endif()

include(sugar_files)

sugar_files(
    BENCHMARK_HEADERS
    caffe_reader.hpp
    io.hpp
    mxnet_ndarray_reader.hpp
)

sugar_files(
    BENCHMARK_SOURCES
    caffe_reader.cpp
    mxnet_ndarray_reader.cpp
)

