#pragma once

// https://github.com/apache/incubator-mxnet/blob/6c5162915655d7df93c8598b345a6bdfd053e600/src/ndarray/ndarray.cc#L849

/* magic number for ndarray version 1, with int64_t TShape */
static const uint32_t NDARRAY_V1_MAGIC = 0xF993fac8;

/* magic number for ndarray version 2, with storage type */
static const uint32_t NDARRAY_V2_MAGIC = 0xF993fac9;

// https://github.com/apache/incubator-mxnet/blob/6c5162915655d7df93c8598b345a6bdfd053e600/src/ndarray/ndarray.cc#L962
