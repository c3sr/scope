#include <benchmark/benchmark.h>

#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <cudnn.h>

#include "init/init.hpp"
#include "utils/utils.hpp"

#define checkCUDNN(expression)                                                                                         \
  {                                                                                                                    \
    cudnnStatus_t status = (expression);                                                                               \
    if (status != CUDNN_STATUS_SUCCESS) {                                                                              \
      std::cerr << "Error on line " << __LINE__ << ": " << cudnnGetErrorString(status) << std::endl;                   \
      std::exit(EXIT_FAILURE);                                                                                         \
    }                                                                                                                  \
  }

// http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
static void CUDNN_CONV(benchmark::State& state) {
  if (!has_cuda) {
    state.SkipWithError("CUDNN/CONV no CUDA device found");
    return;
  }

  const auto batch_size = 1, channels = 3;
  const float alpha = 1, beta = 0;
  cudnnConvolutionMode_t conv_mode = CUDNN_CONVOLUTION;

  // http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionFwdAlgo_t
  cudnnConvolutionFwdAlgo_t convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;

  const auto height        = state.range(0);
  const auto width         = state.range(1);
  const auto kernel_height = state.range(2);
  const auto kernel_width  = kernel_height;

  auto image      = std::vector<float>(batch_size * channels * height * width);
  int image_bytes = batch_size * channels * height * width * sizeof(float);
  std::fill(image.begin(), image.end(), 1.0f);

  auto kernel      = std::vector<float>(batch_size * channels * kernel_height * kernel_width);
  int kernel_bytes = batch_size * channels * kernel_height * kernel_width * sizeof(float);
  std::fill(kernel.begin(), kernel.end(), 1.0f);

  cudnnHandle_t cudnn_handle;
  checkCUDNN(cudnnCreate(&cudnn_handle));

  if (PRINT_IF_ERROR(cudnnCreate(&cudnn_handle))) {
    LOG(critical, "CUDNN/CONV initialization failed");
    state.SkipWithError("CUDNN/CONV initialization failed");
    return;
  }
  defer(cudnnDestroy(cudnn_handle));

  cudnnTensorDescriptor_t input_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                        /*format=*/CUDNN_TENSOR_NHWC,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/batch_size,
                                        /*channels=*/channels,
                                        /*image_height=*/height,
                                        /*image_width=*/width));
  defer(cudnnDestroyTensorDescriptor(input_descriptor));

  cudnnTensorDescriptor_t output_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                        /*format=*/CUDNN_TENSOR_NHWC,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/batch_size,
                                        /*channels=*/channels,
                                        /*image_height=*/height,
                                        /*image_width=*/width));
  defer(cudnnDestroyTensorDescriptor(output_descriptor));

  cudnnFilterDescriptor_t kernel_descriptor;
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*out_channels=*/channels,
                                        /*in_channels=*/channels,
                                        /*kernel_height=*/kernel_height,
                                        /*kernel_width=*/kernel_width));
  defer(cudnnDestroyFilterDescriptor(kernel_descriptor));

  cudnnConvolutionDescriptor_t convolution_descriptor;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                             /*pad_height=*/1,
                                             /*pad_width=*/1,
                                             /*vertical_stride=*/1,
                                             /*horizontal_stride=*/1,
                                             /*dilation_height=*/1,
                                             /*dilation_width=*/1,
                                             /*mode=*/conv_mode,
                                             /*computeType=*/CUDNN_DATA_FLOAT));
  defer(cudnnDestroyConvolutionDescriptor(convolution_descriptor));

  size_t workspace_bytes = 0;
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                                                     input_descriptor,
                                                     kernel_descriptor,
                                                     convolution_descriptor,
                                                     output_descriptor,
                                                     convolution_algorithm,
                                                     &workspace_bytes));
  // std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB" << std::endl;

  void* d_workspace{nullptr};
  if (PRINT_IF_ERROR(cudaMalloc(&d_workspace, workspace_bytes))) {
    LOG(critical, "CUDNN/CONV device memory allocation failed for workspace");
    state.SkipWithError("CUDNN/CONV device memory allocation failed for workspace");
    return;
  }
  defer(cudaFree(d_workspace));

  float* d_input{nullptr};
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

  float* d_output{nullptr};
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

  float* d_kernel{nullptr};
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
      state.SkipWithError("CUDNN/CONV failed to launch kernel");
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
                         {"kernel_height", kernel_height},
                         {"Flops", {2.0 * height * width * kernel_height, benchmark::Counter::kAvgThreadsRate}}});
  // state.SetBytesProcessed(int64_t(state.iterations()) * a.size() * b.size() * c.size());
  state.SetItemsProcessed(int64_t(state.iterations()) * height * width * kernel_height);
}

#ifdef USE_CUDA_EVENTS
BENCHMARK(CUDNN_CONV)->ALL_ARGS()->UseManualTime();
#else  // USE_CUDA_EVENTS
BENCHMARK(CUDNN_CONV)->ALL_ARGS();
#endif // USE_CUDA_EVENTS
