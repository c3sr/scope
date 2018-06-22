#define BENCHMARK_NAME "CUDNN/CONV"

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

#include "layer/args.hpp"
#include "layer/helper.hpp"
#include "layer/utils.hpp"

// Calculates convolution output dimension using the definition from Caffe
static inline int calc_out_dim(int input_dim, int filter_dim, int padd, int stride) {
  return (input_dim - filter_dim + 2 * padd) / stride + 1;
}

// http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
// http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionFwdAlgo_t
template <typename T, cudnnConvolutionFwdAlgo_t convolution_algorithm, cudnnMathType_t math_type = CUDNN_DEFAULT_MATH>
static void CUDNN_Impl(benchmark::State& state,
                       std::string convolution_algorithm_name        = "",
                       std::string convolution_algorithm_description = "") {
  if (!has_cuda) {
    state.SkipWithError(BENCHMARK_NAME " no CUDA device found");
    return;
  }
  if (math_type == CUDNN_TENSOR_OP_MATH && !detail::SupportsTensorCore(FLAG(cuda_device_id))) {
    state.SkipWithError(BENCHMARK_NAME "no Tensorcore support on current device");
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

  // const auto output_width  = calc_out_dim(width, filter_width, pad_width, stride_width);
  // const auto output_height = calc_out_dim(height, filter_height.fh, pad_height, stride_height);
  const auto format = std::is_integral<T>::value ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;

  auto input_image = std::vector<T>(input_image_bytes / sizeof(T));
  auto kernel      = std::vector<T>(kernel_bytes / sizeof(T));

  std::fill(input_image.begin(), input_image.end(), detail::one<T>());
  std::fill(kernel.begin(), kernel.end(), detail::one<T>());

  auto input_tensor = Tensor<T>(state,
                                {/*batch_size=*/batch_size,
                                 /*channels=*/channels,
                                 /*image_height=*/height,
                                 /*image_width=*/width});
  if (!input_tensor.is_valid) {
    return;
  }
  cudnnTensorDescriptor_t input_descriptor = input_tensor.get();

  const auto kernel_filter = Filter<T>(state,
                                       {/*out_channels=*/kernel_size,
                                        /*in_channels=*/channels,
                                        /*kernel_height=*/filter_height,
                                        /*kernel_width=*/filter_width});
  if (!kernel_filter.is_valid) {
    return;
  }
  cudnnFilterDescriptor_t kernel_descriptor = kernel_filter.get();

  cudnnConvolutionDescriptor_t convolution_descriptor;
  if (PRINT_IF_ERROR(cudnnCreateConvolutionDescriptor(&convolution_descriptor))) {
    state.SkipWithError(BENCHMARK_NAME " failed to cudnnCreateConvolutionDescriptor");
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
    state.SkipWithError(BENCHMARK_NAME " failed to cudnnSetConvolution2dDescriptor");
    return;
  }
  defer(cudnnDestroyConvolutionDescriptor(convolution_descriptor));

  cudnnSetConvolutionMathType(convolution_descriptor, math_type);

  int out_h, out_w, out_c, out_n;

  if (PRINT_IF_ERROR(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor, input_descriptor, kernel_descriptor,
                                                           &out_n, &out_c, &out_h, &out_w))) {
    state.SkipWithError(BENCHMARK_NAME " failed to cudnnGetConvolution2dForwardOutputDim");
    return;
  }

  auto output_tensor = Tensor<T>(state,
                                 {/*batch_size=*/out_n,
                                  /*channels=*/out_c,
                                  /*image_height=*/out_h,
                                  /*image_width=*/out_w});
  if (!output_tensor.is_valid) {
    return;
  }
  cudnnTensorDescriptor_t output_descriptor = output_tensor.get();

  const auto output_image_bytes = sizeof(T) * out_n * out_c * out_h * out_w;

  size_t workspace_bytes = 0;

  cudnnConvolutionFwdAlgo_t advised_convolution_algorithm;
  if (PRINT_IF_ERROR(cudnnGetConvolutionForwardAlgorithm(
          cudnn_handle, input_descriptor, kernel_descriptor, convolution_descriptor, output_descriptor,
          CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &advised_convolution_algorithm))) {
    advised_convolution_algorithm = (cudnnConvolutionFwdAlgo_t) -1;
  }

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
      state.SkipWithError(BENCHMARK_NAME " failed to cudnnGetConvolutionForwardWorkspaceSize");
      return;
    }
  }
  // std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB" << std::endl;

  void* d_workspace{nullptr};
  if (PRINT_IF_ERROR(cudaMalloc(&d_workspace, workspace_bytes))) {
    LOG(critical, BENCHMARK_NAME " device memory allocation failed for workspace");
    state.SkipWithError(BENCHMARK_NAME " device memory allocation failed for workspace");
    return;
  }
  defer(cudaFree(d_workspace));

  T* d_input{nullptr};
  if (PRINT_IF_ERROR(cudaMalloc(&d_input, input_image_bytes))) {
    LOG(critical, BENCHMARK_NAME " device memory allocation failed for input");
    state.SkipWithError(BENCHMARK_NAME " device memory allocation failed for output");
    return;
  }
  defer(cudaFree(d_input));

  if (PRINT_IF_ERROR(cudaMemcpy(d_input, input_image.data(), input_image_bytes, cudaMemcpyHostToDevice))) {
    LOG(critical, BENCHMARK_NAME " failed to copy image vector to device");
    state.SkipWithError(BENCHMARK_NAME " failed to copy image vector to device");
    return;
  }

  T* d_output{nullptr};
  if (PRINT_IF_ERROR(cudaMalloc(&d_output, output_image_bytes))) {
    LOG(critical, BENCHMARK_NAME " device memory allocation failed for input");
    state.SkipWithError(BENCHMARK_NAME " device memory allocation failed for output");
    return;
  }
  defer(cudaFree(d_output));

  if (PRINT_IF_ERROR(cudaMemset(d_output, 0, output_image_bytes))) {
    LOG(critical, BENCHMARK_NAME " failed to initialize output to 0 on device");
    state.SkipWithError(BENCHMARK_NAME " failed initialize output to 0 on device");
    return;
  }

  T* d_kernel{nullptr};
  if (PRINT_IF_ERROR(cudaMalloc(&d_kernel, kernel_bytes))) {
    LOG(critical, BENCHMARK_NAME " device memory allocation failed for input");
    state.SkipWithError(BENCHMARK_NAME " device memory allocation failed for output");
    return;
  }
  defer(cudaFree(d_kernel));

  if (PRINT_IF_ERROR(cudaMemcpy(d_kernel, kernel.data(), sizeof(kernel_bytes), cudaMemcpyHostToDevice))) {
    LOG(critical, BENCHMARK_NAME " failed to copy kernel vector to device");
    state.SkipWithError(BENCHMARK_NAME " failed to copy kernel vector to device");
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
      state.SkipWithError(BENCHMARK_NAME " failed to perform cudnnConvolutionForward");
      break;
    }
    if (PRINT_IF_ERROR(cuda_err)) {
      state.SkipWithError(BENCHMARK_NAME " failed to launch kernel");
      break;
    }

    float msecTotal = 0.0f;
    if (PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop))) {
      state.SkipWithError(BENCHMARK_NAME " failed to launch kernel");
      break;
    }
    state.SetIterationTime(msecTotal / 1000);
    state.ResumeTiming();
  }

  const auto P = out_h, Q = out_w;

  const auto compute_flops = [&](cudnnConvolutionFwdAlgo_t alg) {
    switch (alg) {
      case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
      case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
      case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
        // flops = 2 * filter_width * filter_height * out_w * out_h * channels * out_c * batch_size *
        // state.iterations(); 2KCRSNPQ
        return static_cast<double>(2) * K * C * R * S * N * P * Q;
      case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
      case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
        //(NCKHW + (NC +CK +NK)HW log(HW))
        return static_cast<double>(N * C * K * H * W +
                                   (N * C + C * K + N * K) * (H * W) * log(static_cast<double>(H * W)));
      case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
      case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
        return static_cast<double>(-1); // todo ... implement
      default:
        return static_cast<double>(-1);
    }
  };

  const double predicted_flops         = compute_flops(convolution_algorithm);
  const double predicted_advised_flops = compute_flops(advised_convolution_algorithm);

  state.counters.insert(
      {{"input_height", height},
       {"input_width", width},
       {"input_channels", channels},
       {"input_batch_size", batch_size},
       {"output_height", out_h},
       {"output_width", out_w},
       {"output_channels", out_c},
       {"output_batch_size", out_n},
       {"filter_height", filter_height},
       {"filter_width", filter_width},
       {"pad_height", pad_height},
       {"pad_width", pad_width},
       {"stride_height", stride_height},
       {"stride_width", stride_width},
       {"workspace_bytes", workspace_bytes},
       {"workspace_megabytes", workspace_bytes / 1048576.0},
       {"convolution_algorithm", convolution_algorithm},
       {"advised_convolution_algorithm", advised_convolution_algorithm},
       {"math_type", (int) math_type},
       {"predicted_flops", {predicted_flops * state.iterations(), benchmark::Counter::kAvgThreadsRate}},
       {"predicted_advised_flops",
        {predicted_advised_flops * state.iterations(), benchmark::Counter::kAvgThreadsRate}}});
  state.SetItemsProcessed(int64_t(state.iterations()) * N * K * C * W * H);
}

// template <typename T>
// static void CUDNN(benchmark::State& state) {
//   CUDNN_Impl<T, CUDNN_CONVOLUTION_FWD_ALGO_GEMM>(
//       state,
//       "CUDNN_CONVOLUTION_FWD_ALGO_GEMM",
//       "This algorithm expresses the convolution as an explicit matrix product. A "
//       "significant memory workspace is needed to store the matrix that holds the "
//       "input tensor data.");
//   CUDNN_Impl<T, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM>(
//       state,
//       "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
//       "This algorithm expresses the convolution as a matrix product without actually explicitly form the matrix that
//       " "holds the input tensor data, but still needs some memory workspace to precompute some indices in order to "
//       "facilitate the implicit construction of the matrix that holds the input tensor data.");
//   if (std::is_same<T, int8_t>::value) {
//     return;
//   }
//   CUDNN_Impl<T, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM>(
//       state,
//       "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",
//       "This algorithm expresses the convolution as a matrix product "
//       "without actually explicitly form the matrix that holds the input "
//       "tensor data.");
// #if 0
//   CUDNN_Impl<T, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT>(
//       state,
//       "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",
//       "This algorithm expresses the convolution as a direct convolution (e.g "
//       "without implicitly or explicitly doing a matrix multiplication).");
//   CUDNN_Impl<T, CUDNN_CONVOLUTION_FWD_ALGO_FFT>(
//       state,
//       "CUDNN_CONVOLUTION_FWD_ALGO_FFT",
//       "This algorithm uses the Fast-Fourier Transform approach to compute the "
//       "convolution. A significant memory workspace is needed to store "
//       "intermediate results.");
//   CUDNN_Impl<T, CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING>(
//       state,
//       "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
//       "This algorithm uses the Fast-Fourier Transform approach but splits "
//       "the inputs into tiles. A significant memory workspace is needed to "
//       "store intermediate results but less than "
//       "CUDNN_CONVOLUTION_FWD_ALGO_FFT for large size images.");
//   CUDNN_Impl<T, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD>(
//       state,
//       "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",
//       "This algorithm uses the Winograd Transform approach to compute the "
//       "convolution. A reasonably sized workspace is needed to store "
//       "intermediate results.");
//   CUDNN_Impl<T, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED>(state,
//                                                               "CUDNN_CONVOLUTION_FWD_ALGO_​WINOGRAD_NONFUSED",
//                                                               "This algorithm uses the Winograd Transform approach to
//                                                               " "compute the convolution. Significant workspace may
//                                                               be " "needed to store intermediate results.");
// #endif
// }

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FORWARD_INT8(benchmark::State& state) {
  CUDNN_Impl<int8_t, convolution_algorithm>(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FORWARD_INT32(benchmark::State& state) {
  CUDNN_Impl<int32_t, convolution_algorithm>(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FORWARD_HALF(benchmark::State& state) {
  CUDNN_Impl<__half, convolution_algorithm>(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FORWARD_HALF_TENSOROP(benchmark::State& state) {
  CUDNN_Impl<__half, convolution_algorithm, CUDNN_TENSOR_OP_MATH>(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FORWARD_FLOAT(benchmark::State& state) {
  CUDNN_Impl<float, convolution_algorithm>(state);
}

template <cudnnConvolutionFwdAlgo_t convolution_algorithm>
static void LAYER_CUDNN_CONV_FORWARD_DOUBLE(benchmark::State& state) {
  CUDNN_Impl<double, convolution_algorithm>(state);
}

#define CONV_PROBLEMS ALL_CONV_PROBLEMS

#define BENCHMARK_CUDNN(b)                                                                                             \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)->CONV_PROBLEMS()->UseManualTime();                   \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)->CONV_PROBLEMS()->UseManualTime();           \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_GEMM)->CONV_PROBLEMS()->UseManualTime();                            \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT)->CONV_PROBLEMS()->UseManualTime();                          \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_FFT)->CONV_PROBLEMS()->UseManualTime();                             \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING)->CONV_PROBLEMS()->UseManualTime();                      \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)->CONV_PROBLEMS()->UseManualTime();                        \
  BENCHMARK_TEMPLATE(b, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED)->CONV_PROBLEMS()->UseManualTime()

BENCHMARK_CUDNN(LAYER_CUDNN_CONV_FORWARD_INT8);
BENCHMARK_CUDNN(LAYER_CUDNN_CONV_FORWARD_INT32);
BENCHMARK_CUDNN(LAYER_CUDNN_CONV_FORWARD_HALF);
BENCHMARK_CUDNN(LAYER_CUDNN_CONV_FORWARD_HALF_TENSOROP);
BENCHMARK_CUDNN(LAYER_CUDNN_CONV_FORWARD_FLOAT);
BENCHMARK_CUDNN(LAYER_CUDNN_CONV_FORWARD_DOUBLE);
