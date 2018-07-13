Benchmark cuDNN for differnt DL layers.

Currently available:

# Activation

* [CUDNN_ACTIVATION_BWD](activation_bwd.cpp)
* [CUDNN_ACTIVATION_FWD](activation_fwd.cpp)

[cudnnActivationMode_t](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnActivationMode_t)

[cudnnActivationBackward](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnActivationBackward)

[cudnnActivationForward](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnActivationForward)

# Batchnorm

* [CUDNN_BATCHNORM_BWD](batchnorm_bwd.cpp)
* [CUDNN_BATCHNORM_FWD_INFERENCE](batchnorm_fwd.cpp)
* [CUDNN_BATCHNORM_FWD_TRAINING](batchnorm_fwd.cpp)

[cudnnBatchNormMode_t](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnBatchNormMode_t)

[cudnnBatchNormalizationBackward](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnBatchNormalizationBackward)

[cudnnBatchNormalizationForwardTraining](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnBatchNormalizationForwardTraining)

[cudnnBatchNormalizationForwardInference](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnBatchNormalizationForwardInference)

# Convoluation

* [CUDNN_CONV_BIAS_ACTIVATION_FWD](conv_bias_activation_fwd.cpp)
* [CUDNN_CONV_BWD_BIAS](conv_bwd_bias.cpp)
* [CUDNN_CONV_BWD_DATA](conv_bwd_data.cpp)
* [CUDNN_CONV_BWD_FILTER](conv_bwd_filter.cpp)
* [CUDNN_CONV_FWD](conv_fwd.cpp)

[cudnnConvolutionBiasActivationForward](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBiasActivationForward)

[cudnnConvolutionBackwardBias](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardBias)

[cudnnConvolutionBwdDataAlgo_t](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBwdDataAlgo_t)

[cudnnConvolutionBackwardData](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardData)

[cudnnConvolutionBwdFilterAlgo_t](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBwdFilterAlgo_t)

[cudnnConvolutionBackwardFilter](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardFilter)


[cudnnConvolutionFwdAlgo_t](http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionFwdAlgo_t)

[cudnnConvolutionForward](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionForward)

# Dropout

* [CUDNN_DROPOUT_BWD](dropout_bwd.cpp)
* [CUDNN_DROPOUT_FWD](dropout_fwd.cpp)

[cudnnDropoutBackward](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnDropoutBackward)

[cudnnDropoutForward](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnDropoutForward)

[cudnnDropoutGetReserveSpaceSize](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnDropoutGetReserveSpaceSize)

# Pooling

* [CUDNN_POOLING_BWD](pooling_bwd.cpp)
* [CUDNN_POOLING_FWD](pooling_fwd.cpp)

[cudnnPoolingBackward](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnPoolingBackward)

[cudnnPoolingForward](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnPoolingForward)

[cudnnGetPooling2dForwardOutputDim](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnGetPooling2dForwardOutputDim)

[cudnnSetPooling2dDescriptor](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnSetPooling2dDescriptor)

# Softmax

* [CUDNN_SOFTMAX_BWD](softmax_bwd.cpp)
* [CUDNN_SOFTMAX_FWD](softmax_fwd.cpp)
  
[cudnnSoftmaxMode_t](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnSoftmaxMode_t)

[cudnnSoftmaxBackward](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnSoftmaxBackward)

[cudnnSoftmaxForward](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnSoftmaxForward)
