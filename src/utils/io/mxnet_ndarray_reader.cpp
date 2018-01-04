
#include <stdint.h>

// https://github.com/apache/incubator-mxnet/blob/6c5162915655d7df93c8598b345a6bdfd053e600/src/ndarray/ndarray.cc#L849

/* magic number for ndarray version 1, with int64_t TShape */
static constexpr uint32_t NDARRAY_V1_MAGIC = 0xF993fac8;

/* magic number for ndarray version 2, with storage type */
static constexpr uint32_t NDARRAY_V2_MAGIC = 0xF993fac9;

// https://github.com/apache/incubator-mxnet/blob/6c5162915655d7df93c8598b345a6bdfd053e600/src/ndarray/ndarray.cc#L962

// https://github.com/dmlc/tvm/blob/8a3dbd7971a3d7df7c3a728d18c7c6124bf68989/src/runtime/graph/graph_runtime.cc#L356
/*! \brief Magic number for NDArray file */
constexpr uint64_t kTVMNDArrayMagic = 0xDD5E40F096B4A13F;
/*! \brief Magic number for NDArray list file  */
constexpr uint64_t kTVMNDArrayListMagic = 0xF7E58D4F05049CB7;
