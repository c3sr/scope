#include <benchmark/benchmark.h>

#include <initializer_list>
#include <iostream>
#include <mutex>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <cudnn.h>

#include "init/init.hpp"
#include "utils/utils.hpp"

#include "layer/utils.hpp"

#ifndef BENCHMARK_NAME
#define BENCHMARK_NAME "CUDNN"
#endif // BENCHMARK_NAME

template <typename T>
struct Filter {
  using type       = T;
  using value_type = valueDataType<T>::type;

  const auto layout = std::is_integral<T>::value ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;

  bool is_valid{false};
  cudnnFilterDescriptor_t descriptor{nullptr};

  Filter(benchmark::State& state, const std::initializer_list<int>& shape) {

    assert(shape.size() <= 4);
    int dims[4] = {1, 1, 1, 1};
    for (int ii = 0; ii < shape.size(); ++ii) {
      dims[ii] = shape[ii];
    }
    if (PRINT_IF_ERROR(cudnnCreateFilterDescriptor(&descriptor))) {
      state.SkipWithError(BENCHMARK_NAME " failed to cudnnCreateFilterDescriptor");
      return;
    }

    if (PRINT_IF_ERROR(cudnnSetFilter4dDescriptor(descriptor, value_type, layout,
      dims[0], dims[1], dims[2], dims[3])) {
      state.SkipWithError(BENCHMARK_NAME " failed to cudnnSetFilter4dDescriptor");
      return;
      }
      is_valid = true;
  }

  ~Filter() {
    if (!is_valid) {
      return;
    }
    PRINT_IF_ERROR(cudnnDestroyFilterDescriptor(descriptor));
  }

  cudnnFilterDescriptor_t get() const {
    if (!is_valid) {
      return nullptr;
    }
    return descriptor;
  }
};

template <typename T>
struct Tensor {
  using type        = T;
  using value_type  = valueDataType<T>::type;
  const auto layout = std::is_integral<T>::value ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;

  bool is_valid{false};
  cudnnTensorDescriptor_t descriptor{nullptr};

  Tensor(benchmark::State& state, const std::initializer_list<int>& shape) {

    assert(shape.size() <= 4);
    int dims[4] = {1, 1, 1, 1};
    for (int ii = 0; ii < shape.size(); ++ii) {
      dims[ii] = shape[ii];
    }
    if (PRINT_IF_ERROR(cudnnCreateTensorDescriptor(&descriptor))) {
      state.SkipWithError(BENCHMARK_NAME " failed to cudnnCreateTensorDescriptor");
      return;
    }

    if (PRINT_IF_ERROR(cudnnSetTensor4dDescriptor(descriptor, layout, value_type,
      dims[0], dims[1], dims[2], dims[3])) {
      state.SkipWithError(BENCHMARK_NAME " failed to cudnnSetTensor4dDescriptor");
      return;
      }
      is_valid = true;
  }

  ~Tensor() {
    if (!is_valid) {
      return;
    }
    PRINT_IF_ERROR(cudnnDestroyTensorDescriptor(descriptor));
  }

  cudnnTensorDescriptor_t get() const {
    if (!is_valid) {
      return nullptr;
    }
    return descriptor;
  }
};
