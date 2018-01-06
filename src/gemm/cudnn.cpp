#include <benchmark/benchmark.h>

#include <iostream>
#include <mutex>
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
  static const cudnnDataType_t type = CUDNN_DATA_INT32;
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

// Calculates convolution output dimension using the definition from Caffe
static inline int calc_out_dim(int input_dim, int filter_dim, int padd, int stride) {
  return (input_dim - filter_dim + 2 * padd) / stride + 1;
}

// http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
// http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionFwdAlgo_t
template <typename T, cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void CUDNN_Impl(benchmark::State& state,
                       std::string convolution_algorithm_name,
                       std::string convolution_algorithm_description) {
  if (!has_cuda) {
    state.SkipWithError("CUDNN/CONV no CUDA device found");
    return;
  }

  (void) convolution_algorithm_name;
  (void) convolution_algorithm_description;

  const float alpha = 1, beta = 0;
  const cudnnConvolutionMode_t conv_mode = CUDNN_CONVOLUTION;

  //  w, h, c, n, k, filter_w(s), filter_h(r), pad_w, pad_h, wstride, hstride
  const auto width         = state.range(0);
  const auto height        = state.range(1);
  const auto channels      = state.range(2);
  const auto batch_size    = state.range(3);
  const auto kernel_size   = state.range(4);
  const auto filter_width  = state.range(5);
  const auto filter_height = state.range(6);
  const auto pad_width     = state.range(7);
  const auto pad_height    = state.range(8);
  const auto stride_width  = state.range(9);
  const auto stride_height = state.range(10);

  const int input_image_bytes = batch_size * channels * height * width * sizeof(T);
  const int kernel_bytes      = kernel_size * channels * filter_height * filter_width * sizeof(T);

  const auto N = batch_size, K = kernel_size, C = channels, H = height, W = width, R = filter_height, S = filter_width;
  const auto format = std::is_integral<T>::value ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;

  auto input_image = std::vector<T>(input_image_bytes / sizeof(T));
  auto kernel      = std::vector<T>(kernel_bytes / sizeof(T));

  std::fill(input_image.begin(), input_image.end(), gemm::detail::one<T>());
  std::fill(kernel.begin(), kernel.end(), gemm::detail::one<T>());

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
                                                /*format=*/format,
                                                /*dataType=*/valueDataType<T>::type,
                                                /*batch_size=*/batch_size,
                                                /*channels=*/channels,
                                                /*image_height=*/height,
                                                /*image_width=*/width))) {
    state.SkipWithError("CUDNN/CONV failed to cudnnSetTensor4dDescriptor");
    return;
  }
  defer(cudnnDestroyTensorDescriptor(input_descriptor));

  cudnnFilterDescriptor_t kernel_descriptor;
  if (PRINT_IF_ERROR(cudnnCreateFilterDescriptor(&kernel_descriptor))) {
    state.SkipWithError("CUDNN/CONV failed to cudnnCreateFilterDescriptor");
    return;
  }
  if (PRINT_IF_ERROR(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                                /*dataType=*/valueDataType<T>::type,
                                                /*format=*/format,
                                                /*out_channels=*/kernel_size,
                                                /*in_channels=*/channels,
                                                /*kernel_height=*/filter_height,
                                                /*kernel_width=*/filter_width))) {
    const auto err_msg = fmt::format("CUDNN/CONV failed to cudnnSetFilter4dDescriptor with out_channels = {}, "
                                     "in_channels = {}, filter_height = {}, filter_width = {}",
                                     kernel_size,
                                     channels,
                                     filter_height,
                                     filter_width);
    state.SkipWithError(err_msg.c_str());
    return;
  }
  defer(cudnnDestroyFilterDescriptor(kernel_descriptor));

  cudnnConvolutionDescriptor_t convolution_descriptor;
  if (PRINT_IF_ERROR(cudnnCreateConvolutionDescriptor(&convolution_descriptor))) {
    state.SkipWithError("CUDNN/CONV failed to cudnnCreateConvolutionDescriptor");
    return;
  }
  if (PRINT_IF_ERROR(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                                     /*pad_height=*/pad_height,
                                                     /*pad_width=*/pad_width,
                                                     /*vertical_stride=*/stride_height,
                                                     /*horizontal_stride=*/stride_width,
                                                     /*dilation_height=*/1,
                                                     /*dilation_width=*/1,
                                                     /*mode=*/conv_mode,
                                                     /*computeType=*/accumDataType<T>::type))) {
    state.SkipWithError("CUDNN/CONV failed to cudnnSetConvolution2dDescriptor");
    return;
  }
  defer(cudnnDestroyConvolutionDescriptor(convolution_descriptor));

  int out_h, out_w, out_c, out_n;

  if (PRINT_IF_ERROR(cudnnGetConvolution2dForwardOutputDim(
          convolution_descriptor, input_descriptor, kernel_descriptor, &out_n, &out_c, &out_h, &out_w))) {
    state.SkipWithError("CUDNN/CONV failed to cudnnGetConvolution2dForwardOutputDim");
    return;
  }

  cudnnTensorDescriptor_t output_descriptor;
  if (PRINT_IF_ERROR(cudnnCreateTensorDescriptor(&output_descriptor))) {
    state.SkipWithError("CUDNN/CONV failed to cudnnCreateTensorDescriptor");
    return;
  }
  if (PRINT_IF_ERROR(cudnnSetTensor4dDescriptor(output_descriptor,
                                                /*format=*/format,
                                                /*dataType=*/valueDataType<T>::type,
                                                /*batch_size=*/out_n,
                                                /*channels=*/out_c,
                                                /*image_height=*/out_h,
                                                /*image_width=*/out_w))) {
    state.SkipWithError("CUDNN/CONV failed to cudnnSetTensor4dDescriptor");
    return;
  }
  defer(cudnnDestroyTensorDescriptor(output_descriptor));

  const auto output_image_bytes = sizeof(T) * out_n * out_c * out_h * out_w;

  size_t workspace_bytes = 0;

  if (std::is_same<T, int8_t>::value) {

    // Note: cudnn workspace size function doesn't work for INT8_CONFIG
    workspace_bytes = 1073741824;
  } else {
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
  if (PRINT_IF_ERROR(cudaMalloc(&d_input, input_image_bytes))) {
    LOG(critical, "CUDNN/CONV device memory allocation failed for input");
    state.SkipWithError("CUDNN/CONV device memory allocation failed for output");
    return;
  }
  defer(cudaFree(d_input));

  if (PRINT_IF_ERROR(cudaMemcpy(d_input, input_image.data(), input_image_bytes, cudaMemcpyHostToDevice))) {
    LOG(critical, "CUDNN/CONV failed to copy image vector to device");
    state.SkipWithError("CUDNN/CONV failed to copy image vector to device");
    return;
  }

  T* d_output{nullptr};
  if (PRINT_IF_ERROR(cudaMalloc(&d_output, output_image_bytes))) {
    LOG(critical, "CUDNN/CONV device memory allocation failed for input");
    state.SkipWithError("CUDNN/CONV device memory allocation failed for output");
    return;
  }
  defer(cudaFree(d_output));

  if (PRINT_IF_ERROR(cudaMemset(d_output, 0, output_image_bytes))) {
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

  // const int output_width = calc_out_dim(width, filter_width, pad_width, stride_width);
  // const int output_height = calc_out_dim(height, filter_height.fh, pad_height, stride_height);

  const auto flops = 2.0 * filter_width * filter_height * out_w * out_h * channels * out_c * batch_size;

  state.counters.insert({{"input_height", height},
                         {"input_width", width},
                         {"input_channels", channels},
                         {"input_batch_size", batch_size},
                         {"input_height", out_h},
                         {"input_width", out_w},
                         {"input_channels", out_c},
                         {"input_batch_size", out_n},
                         {"filter_height", filter_height},
                         {"filter_width", filter_width},
                         {"pad_height", pad_height},
                         {"pad_width", pad_width},
                         {"stride_height", stride_height},
                         {"stride_width", stride_width},
                         {"workspace_bytes", workspace_bytes},
                         {"workspace_megabytes", workspace_bytes / 1048576.0},
                         {"convolution_algorithm", convolution_algorithm},
                         {"Flops", {flops, benchmark::Counter::kAvgThreadsRate}}});
  state.SetItemsProcessed(int64_t(state.iterations()) * N * K * C * W * H);
}

template <typename T>
static void CUDNN(benchmark::State& state) {
  CUDNN_Impl<T, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM>(
      state,
      "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
      "This algorithm expresses the convolution as a matrix product without actually explicitly form the matrix that "
      "holds the input tensor data, but still needs some memory workspace to precompute some indices in order to "
      "facilitate the implicit construction of the matrix that holds the input tensor data.");
  if (std::is_same<T, int8_t>::value) {
    return;
  }
  CUDNN_Impl<T, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM>(
      state,
      "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",
      "This algorithm expresses the convolution as a matrix product "
      "without actually explicitly form the matrix that holds the input "
      "tensor data.");
  CUDNN_Impl<T, CUDNN_CONVOLUTION_FWD_ALGO_GEMM>(
      state,
      "CUDNN_CONVOLUTION_FWD_ALGO_GEMM",
      "This algorithm expresses the convolution as an explicit matrix product. A "
      "significant memory workspace is needed to store the matrix that holds the "
      "input tensor data.");
  CUDNN_Impl<T, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT>(
      state,
      "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",
      "This algorithm expresses the convolution as a direct convolution (e.g "
      "without implicitly or explicitly doing a matrix multiplication).");
  CUDNN_Impl<T, CUDNN_CONVOLUTION_FWD_ALGO_FFT>(
      state,
      "CUDNN_CONVOLUTION_FWD_ALGO_FFT",
      "This algorithm uses the Fast-Fourier Transform approach to compute the "
      "convolution. A significant memory workspace is needed to store "
      "intermediate results.");
  CUDNN_Impl<T, CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING>(
      state,
      "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
      "This algorithm uses the Fast-Fourier Transform approach but splits "
      "the inputs into tiles. A significant memory workspace is needed to "
      "store intermediate results but less than "
      "CUDNN_CONVOLUTION_FWD_ALGO_FFT for large size images.");
  CUDNN_Impl<T, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD>(
      state,
      "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",
      "This algorithm uses the Winograd Transform approach to compute the "
      "convolution. A reasonably sized workspace is needed to store "
      "intermediate results.");
  CUDNN_Impl<T, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED>(state,
                                                              "CUDNN_CONVOLUTION_FWD_ALGO_​WINOGRAD_NONFUSED",
                                                              "This algorithm uses the Winograd Transform approach to "
                                                              "compute the convolution. Significant workspace may be "
                                                              "needed to store intermediate results.");
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
BENCHMARK(CUDNN_CONV_INT8)->ALL_CONV_PROBLEMS()->UseManualTime();
BENCHMARK(CUDNN_CONV_INT32)->ALL_CONV_PROBLEMS()->UseManualTime();
BENCHMARK(CUDNN_CONV_HALF)->ALL_CONV_PROBLEMS()->UseManualTime();
BENCHMARK(CUDNN_CONV_FLOAT)->ALL_CONV_PROBLEMS()->UseManualTime();
BENCHMARK(CUDNN_CONV_DOUBLE)->ALL_CONV_PROBLEMS()->UseManualTime();
#else  // USE_CUDA_EVENTS
BENCHMARK(CUDNN_CONV_INT8)->ALL_CONV_PROBLEMS();
BENCHMARK(CUDNN_CONV_HALF)->ALL_CONV_PROBLEMS();
BENCHMARK(CUDNN_CONV_FLOAT)->ALL_CONV_PROBLEMS();
BENCHMARK(CUDNN_CONV_DOUBLE)->ALL_CONV_PROBLEMS();
#endif // USE_CUDA_EVENTS
