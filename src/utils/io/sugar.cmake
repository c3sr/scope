# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED SRC_UTILS_IO_SUGAR_CMAKE_)
  return()
else()
  set(SRC_UTILS_IO_SUGAR_CMAKE_ 1)
endif()

include(sugar_files)

sugar_files(
    BENCHMARK_HEADERS
    mxnet_ndarray_reader.hpp
    io.hpp
    caffe_reader.hpp
)

sugar_files(
    BENCHMARK_SOURCES
    mxnet_ndarray_reader.cpp
    caffe_reader.cpp
)

