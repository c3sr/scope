#include <benchmark/benchmark.h>

#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <cudnn.h>

#include "init/init.hpp"
#include "utils/utils.hpp"

#include "gemm/args.hpp"
#include "gemm/utils.hpp"

template <typename T>
struct valueDataType {};

template <>
struct valueDataType<int8_t> {
  static const cudnnDataType_t type = CUDNN_DATA_INT8;
};

template <>
struct valueDataType<int32_t> {
  static const cudnnDataType_t type = CUDNN_DATA_INT32;
};

template <>
struct valueDataType<__half> {
  static const cudnnDataType_t type = CUDNN_DATA_HALF;
};

template <>
struct valueDataType<float> {
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
};

template <>
struct valueDataType<double> {
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
};

template <typename T>
struct accumDataType {};

template <>
struct accumDataType<int8_t> {
  static const cudnnDataType_t type = CUDNN_DATA_INT8;
};

template <>
struct accumDataType<int32_t> {
  static const cudnnDataType_t type = CUDNN_DATA_INT32;
};

template <>
struct accumDataType<__half> {
  static const cudnnDataType_t type = CUDNN_DATA_HALF;
};

template <>
struct accumDataType<float> {
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
};

template <>
struct accumDataType<double> {
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
};

// http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
template <typename T>
static void CUDNN(benchmark::State& state) {
  if (!has_cuda) {
    state.SkipWithError("CUDNN/CONV no CUDA device found");
    return;
  }

  const auto batch_size = 1, channels = 3;
  const float alpha = 1, beta = 0;
  const cudnnConvolutionMode_t conv_mode = CUDNN_CONVOLUTION;

  // http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionFwdAlgo_t
  const cudnnConvolutionFwdAlgo_t convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;

  const auto height        = state.range(0);
  const auto width         = state.range(1);
  const auto kernel_height = state.range(2);
  const auto kernel_width  = kernel_height;

  auto image      = std::vector<T>(batch_size * channels * height * width);
  int image_bytes = batch_size * channels * height * width * sizeof(T);
  std::fill(image.begin(), image.end(), gemm::detail::one<T>());

  auto kernel      = std::vector<T>(batch_size * channels * kernel_height * kernel_width);
  int kernel_bytes = batch_size * channels * kernel_height * kernel_width * sizeof(T);
  std::fill(kernel.begin(), kernel.end(), gemm::detail::one<T>());

  const auto N = batch_size, K = kernel_height, C = channels, I = height, W = width;

  cudnnHandle_t cudnn_handle;

  if (PRINT_IF_ERROR(cudnnCreate(&cudnn_handle))) {
    state.SkipWithError("CUDNN/CONV failed to cudnnCreate");
    return;
  }
  defer(cudnnDestroy(cudnn_handle));

  cudnnTensorDescriptor_t input_descriptor;
  if (PRINT_IF_ERROR(cudnnCreateTensorDescriptor(&input_descriptor))) {
    state.SkipWithError("CUDNN/CONV failed to cudnnCreateTensorDescriptor");
    return;
  }
  if (PRINT_IF_ERROR(cudnnSetTensor4dDescriptor(input_descriptor,
                                                /*format=*/CUDNN_TENSOR_NHWC,
                                                /*valueDataType=*/valueDataType<T>::type,
                                                /*batch_size=*/batch_size,
                                                /*channels=*/channels,
                                                /*image_height=*/height,
                                                /*image_width=*/width))) {
    state.SkipWithError("CUDNN/CONV failed to cudnnSetTensor4dDescriptor");
    return;
  }
  defer(cudnnDestroyTensorDescriptor(input_descriptor));

  cudnnTensorDescriptor_t output_descriptor;
  if (PRINT_IF_ERROR(cudnnCreateTensorDescriptor(&output_descriptor))) {
    state.SkipWithError("CUDNN/CONV failed to cudnnCreateTensorDescriptor");
    return;
  }
  if (PRINT_IF_ERROR(cudnnSetTensor4dDescriptor(output_descriptor,
                                                /*format=*/CUDNN_TENSOR_NHWC,
                                                /*valueDataType=*/valueDataType<T>::type,
                                                /*batch_size=*/batch_size,
                                                /*channels=*/channels,
                                                /*image_height=*/height,
                                                /*image_width=*/width))) {
    state.SkipWithError("CUDNN/CONV failed to cudnnSetTensor4dDescriptor");
    return;
  }
  defer(cudnnDestroyTensorDescriptor(output_descriptor));

  cudnnFilterDescriptor_t kernel_descriptor;
  if (PRINT_IF_ERROR(cudnnCreateFilterDescriptor(&kernel_descriptor))) {
    state.SkipWithError("CUDNN/CONV failed to cudnnCreateFilterDescriptor");
    return;
  }
  if (PRINT_IF_ERROR(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                                /*valueDataType=*/valueDataType<T>::type,
                                                /*format=*/CUDNN_TENSOR_NCHW,
                                                /*out_channels=*/channels,
                                                /*in_channels=*/channels,
                                                /*kernel_height=*/kernel_height,
                                                /*kernel_width=*/kernel_width))) {
    state.SkipWithError("CUDNN/CONV failed to cudnnSetFilter4dDescriptor");
    return;
  }
  defer(cudnnDestroyFilterDescriptor(kernel_descriptor));

  cudnnConvolutionDescriptor_t convolution_descriptor;
  if (PRINT_IF_ERROR(cudnnCreateConvolutionDescriptor(&convolution_descriptor))) {
    state.SkipWithError("CUDNN/CONV failed to cudnnCreateConvolutionDescriptor");
    return;
  }
  if (PRINT_IF_ERROR(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                                     /*pad_height=*/1,
                                                     /*pad_width=*/1,
                                                     /*vertical_stride=*/1,
                                                     /*horizontal_stride=*/1,
                                                     /*dilation_height=*/1,
                                                     /*dilation_width=*/1,
                                                     /*mode=*/conv_mode,
                                                     /*computeType=*/valueDataType<T>::type))) {
    state.SkipWithError("CUDNN/CONV failed to cudnnSetConvolution2dDescriptor");
    return;
  }
  defer(cudnnDestroyConvolutionDescriptor(convolution_descriptor));

  size_t workspace_bytes = 0;
  if (PRINT_IF_ERROR(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                                                             input_descriptor,
                                                             kernel_descriptor,
                                                             convolution_descriptor,
                                                             output_descriptor,
                                                             convolution_algorithm,
                                                             &workspace_bytes))) {
    state.SkipWithError("CUDNN/CONV failed to cudnnGetConvolutionForwardWorkspaceSize");
    return;
  }
  // std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB" << std::endl;

  void* d_workspace{nullptr};
  if (PRINT_IF_ERROR(cudaMalloc(&d_workspace, workspace_bytes))) {
    LOG(critical, "CUDNN/CONV device memory allocation failed for workspace");
    state.SkipWithError("CUDNN/CONV device memory allocation failed for workspace");
    return;
  }
  defer(cudaFree(d_workspace));

  T* d_input{nullptr};
  if (PRINT_IF_ERROR(cudaMalloc(&d_input, image_bytes))) {
    LOG(critical, "CUDNN/CONV device memory allocation failed for input");
    state.SkipWithError("CUDNN/CONV device memory allocation failed for output");
    return;
  }
  defer(cudaFree(d_input));

  if (PRINT_IF_ERROR(cudaMemcpy(d_input, image.data(), image_bytes, cudaMemcpyHostToDevice))) {
    LOG(critical, "CUDNN/CONV failed to copy image vector to device");
    state.SkipWithError("CUDNN/CONV failed to copy image vector to device");
    return;
  }

  T* d_output{nullptr};
  if (PRINT_IF_ERROR(cudaMalloc(&d_output, image_bytes))) {
    LOG(critical, "CUDNN/CONV device memory allocation failed for input");
    state.SkipWithError("CUDNN/CONV device memory allocation failed for output");
    return;
  }
  defer(cudaFree(d_output));

  if (PRINT_IF_ERROR(cudaMemset(d_output, 0, image_bytes))) {
    LOG(critical, "CUDNN/CONV failed to initialize output to 0 on device");
    state.SkipWithError("CUDNN/CONV failed initialize output to 0 on device");
    return;
  }

  T* d_kernel{nullptr};
  if (PRINT_IF_ERROR(cudaMalloc(&d_kernel, kernel_bytes))) {
    LOG(critical, "CUDNN/CONV device memory allocation failed for input");
    state.SkipWithError("CUDNN/CONV device memory allocation failed for output");
    return;
  }
  defer(cudaFree(d_kernel));

  if (PRINT_IF_ERROR(cudaMemcpy(d_kernel, kernel.data(), sizeof(kernel_bytes), cudaMemcpyHostToDevice))) {
    LOG(critical, "CUDNN/CONV failed to copy kernel vector to device");
    state.SkipWithError("CUDNN/CONV failed to copy kernel vector to device");
    return;
  }

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  for (auto _ : state) {
    cudaEventRecord(start, NULL);

    const cudnnStatus_t cudnn_err = cudnnConvolutionForward(cudnn_handle,
                                                            &alpha,
                                                            input_descriptor,
                                                            d_input,
                                                            kernel_descriptor,
                                                            d_kernel,
                                                            convolution_descriptor,
                                                            convolution_algorithm,
                                                            d_workspace,
                                                            workspace_bytes,
                                                            &beta,
                                                            output_descriptor,
                                                            d_output);

    cudaEventRecord(stop, NULL);
    const auto cuda_err = cudaEventSynchronize(stop);

    state.PauseTiming();
    if (PRINT_IF_ERROR(cudnn_err)) {
      state.SkipWithError("CUDNN/CONV failed to perform cudnnConvolutionForward");
      break;
    }
    if (PRINT_IF_ERROR(cuda_err)) {
      state.SkipWithError("CUDNN/CONV failed to launch kernel");
      break;
    }

    float msecTotal = 0.0f;
    if (PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop))) {
      state.SkipWithError("CUDNN/CONV failed to launch kernel");
      break;
    }
    state.SetIterationTime(msecTotal / 1000);
    state.ResumeTiming();
  }

  state.counters.insert({{"height", height},
                         {"width", width},
                         {"kernel_height", kernel_width},
                         {"kernel_height", kernel_height},
                         {"Flops", {2.0 * N * K * C * I * I * W * W, benchmark::Counter::kAvgThreadsRate}}});
  // state.SetBytesProcessed(int64_t(state.iterations()) * a.size() * b.size() * c.size());
  state.SetItemsProcessed(int64_t(state.iterations()) * height * width * kernel_height);
}

static void CUDNN_CONV_INT8(benchmark::State& state) {
  CUDNN<int8_t>(state);
}

static void CUDNN_CONV_INT32(benchmark::State& state) {
  CUDNN<int32_t>(state);
}

static void CUDNN_CONV_HALF(benchmark::State& state) {
  CUDNN<__half>(state);
}

static void CUDNN_CONV_FLOAT(benchmark::State& state) {
  CUDNN<float>(state);
}

static void CUDNN_CONV_DOUBLE(benchmark::State& state) {
  CUDNN<double>(state);
}

#ifdef USE_CUDA_EVENTS
BENCHMARK(CUDNN_CONV_INT8)->ALL_ARGS()->UseManualTime();
BENCHMARK(CUDNN_CONV_INT32)->ALL_ARGS()->UseManualTime();
BENCHMARK(CUDNN_CONV_HALF)->ALL_ARGS()->UseManualTime();
BENCHMARK(CUDNN_CONV_FLOAT)->ALL_ARGS()->UseManualTime();
BENCHMARK(CUDNN_CONV_DOUBLE)->ALL_ARGS()->UseManualTime();
#else  // USE_CUDA_EVENTS
BENCHMARK(CUDNN_CONV_INT8)->ALL_ARGS();
BENCHMARK(CUDNN_CONV_INT32)->ALL_ARGS();
BENCHMARK(CUDNN_CONV_HALF)->ALL_ARGS();
BENCHMARK(CUDNN_CONV_FLOAT)->ALL_ARGS();
BENCHMARK(CUDNN_CONV_DOUBLE)->ALL_ARGS();
#endif // USE_CUDA_EVENTS
