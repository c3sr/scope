#include <benchmark/benchmark.h>

#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <cudnn.h>

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

// http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
int main(int argc, char const *argv[]) {
  cudnnHandle_t cudnn;
  checkCUDNN(cudnnCreate(&cudnn));


  int batch_size = 1, channels = 3, height = 100, width = 10;
  int kernel_height = 3, kernel_width = 3;
cudnnConvolutionMode_t conv_mode = CUDNN_CONVOLUTION;
  auto image = std::vector<float>(batch_size * channels * height * width);
  int image_bytes = batch_size * channels * height * width * sizeof(float);

  std::fill(a.begin(), a.end(), one);


// http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionFwdAlgo_t
cudnnConvolutionFwdAlgo_t convolution_algorithm= CUDNN_CONVOLUTION_FWD_ALGO_GEMM;

  cudnnTensorDescriptor_t input_descriptor;
checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/batch_size,
                                      /*channels=*/channels,
                                      /*image_height=*/height,
                                      /*image_width=*/width));

cudnnTensorDescriptor_t output_descriptor;
checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/batch_size,
                                      /*channels=*/channels,
                                      /*image_height=*/height,
                                      /*image_width=*/width));

cudnnFilterDescriptor_t kernel_descriptor;
checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*out_channels=*/channels,
                                      /*in_channels=*/channels,
                                      /*kernel_height=*/kernel_height,
                                      /*kernel_width=*/kernel_width));

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

size_t workspace_bytes = 0;
checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                   input_descriptor,
                                                   kernel_descriptor,
                                                   convolution_descriptor,
                                                   output_descriptor,
                                                   convolution_algorithm,
                                                   &workspace_bytes));
std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
          << std::endl;

void* d_workspace{nullptr};
cudaMalloc(&d_workspace, workspace_bytes);

float* d_input{nullptr};
cudaMalloc(&d_input, image_bytes);
cudaMemcpy(d_input, image.data(), image_bytes, cudaMemcpyHostToDevice);

float* d_output{nullptr};
cudaMalloc(&d_output, image_bytes);
cudaMemset(d_output, 0, image_bytes);

// Mystery kernel
const float kernel_template[3][3] = {
  {1,  1, 1},
  {1, -8, 1},
  {1,  1, 1}
};

float h_kernel[3][3][3][3];
for (int kernel = 0; kernel < 3; ++kernel) {
  for (int channel = 0; channel < 3; ++channel) {
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        h_kernel[kernel][channel][row][column] = kernel_template[row][column];
      }
    }
  }
}

float* d_kernel{nullptr};
cudaMalloc(&d_kernel, sizeof(h_kernel));
cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

const float alpha = 1, beta = 0;
checkCUDNN(cudnnConvolutionForward(cudnn,
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
                                   d_output));

float* h_output = new float[image_bytes];
cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost);

// Do something with h_output ...

delete[] h_output;
cudaFree(d_kernel);
cudaFree(d_input);
cudaFree(d_output);
cudaFree(d_workspace);

cudnnDestroyTensorDescriptor(input_descriptor);
cudnnDestroyTensorDescriptor(output_descriptor);
cudnnDestroyFilterDescriptor(kernel_descriptor);
cudnnDestroyConvolutionDescriptor(convolution_descriptor);

cudnnDestroy(cudnn);

}
