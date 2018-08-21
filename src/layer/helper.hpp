#pragma once

#include <benchmark/benchmark.h>

#include <initializer_list>
#include <iostream>
#include <mutex>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include <cudnn.h>

#include "init/init.hpp"
#include "utils/utils.hpp"

#include "layer/utils.hpp"

#ifndef BENCHMARK_NAME
#define BENCHMARK_NAME "CUDNN"
#endif // BENCHMARK_NAME

template <typename T>
struct DeviceMemory {
  using type = T;
  T *ptr{nullptr};
  bool is_valid{false};
  size_t size;
  DeviceMemory(benchmark::State &state, const size_t &size) : size(size) {
    if (PRINT_IF_ERROR(cudaMalloc(&ptr, size))) {
      state.SkipWithError(BENCHMARK_NAME " device memory allocation failed");
      return;
    }
    if (PRINT_IF_ERROR(cudaMemset(ptr, 0, size))) {
      state.SkipWithError(BENCHMARK_NAME " device memory set failed");
      return;
    }
    is_valid = true;
  }
  DeviceMemory(benchmark::State &state, const T *data, const size_t &size) : size(size) {
    if (PRINT_IF_ERROR(cudaMalloc(&ptr, size))) {
      state.SkipWithError(BENCHMARK_NAME " device memory allocation failed");
      return;
    }
    if (PRINT_IF_ERROR(cudaMemcpy(ptr, data, size, cudaMemcpyHostToDevice))) {
      state.SkipWithError(BENCHMARK_NAME " device memory copy failed");
      return;
    }
    is_valid = true;
  }
  ~DeviceMemory() {
    if (ptr == nullptr) {
      return;
    }
    cudaFree(ptr);
  }
  T *get() {
    return ptr;
  }
};

template <typename T>
struct Filter {
  using type                   = T;
  static const auto value_type = valueDataType<T>::type;

  static const auto layout = std::is_integral<T>::value ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;
  std::vector<int> shape{};

  bool is_valid{false};
  cudnnFilterDescriptor_t descriptor{nullptr};

  Filter(benchmark::State &state, const std::initializer_list<int> &shape0) : shape(shape0) {

    assert(shape.size() <= 4);
    int dims[4] = {1, 1, 1, 1};
    for (size_t ii = 0; ii < shape.size(); ++ii) {
      dims[ii] = shape[ii];
    }
    if (PRINT_IF_ERROR(cudnnCreateFilterDescriptor(&descriptor))) {
      state.SkipWithError(BENCHMARK_NAME " failed to cudnnCreateFilterDescriptor");
      return;
    }

    if (PRINT_IF_ERROR(
            cudnnSetFilter4dDescriptor(descriptor, value_type, layout, dims[0], dims[1], dims[2], dims[3]))) {
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
  using type                   = T;
  static const auto value_type = valueDataType<T>::type;
  static const auto layout     = std::is_integral<T>::value ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;
  std::vector<int> shape{};

  bool is_valid{false};
  cudnnTensorDescriptor_t descriptor{nullptr};

  Tensor(benchmark::State &state, const std::initializer_list<int> &shape0) : shape(shape0) {

    assert(shape.size() <= 4);
    int dims[4] = {1, 1, 1, 1};
    for (size_t ii = 0; ii < shape.size(); ++ii) {
      dims[ii] = shape[ii];
    }
    if (PRINT_IF_ERROR(cudnnCreateTensorDescriptor(&descriptor))) {
      state.SkipWithError(BENCHMARK_NAME " failed to cudnnCreateTensorDescriptor");
      return;
    }

    if (PRINT_IF_ERROR(
            cudnnSetTensor4dDescriptor(descriptor, layout, value_type, dims[0], dims[1], dims[2], dims[3]))) {
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

template <typename T>
struct Layer {
  using type                   = T;
  static const auto value_type = valueDataType<T>::type;

  std::vector<int> input_shape{}, output_shape{};

  virtual std::vector<int> configure(const std::vector<int> &shape) = 0;

  virtual std::string to_string() const = 0;

  virtual Tensor<float> forward(const Tensor<float> &x) const = 0;

  virtual Tensor<float> backward(const Tensor<T> &dy, const Tensor<T> &y, const Tensor<T> &x,
                                 bool lastLayer = false) = 0;

  virtual size_t predicted_flops() const         = 0;
  virtual size_t predicted_advised_flops() const = 0;
};

// class Flatten : public Layer {

// class Dropout : public Layer {

// class Pooling : public Layer {

// class LRN : public Layer {

// class Activation : public Layer {

// class Softmax : public Layer {
