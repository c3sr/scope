#! /bin/bash

mkdir -p build
pushd build
make -j

CUDA_DEVICE=1
OUT_DIR="../results/cudnn/"`hostname`

echo $OUT_DIR
mkdir -p $OUT_DIR


cudnn_bmarks=(
  CUDNN_ACTIVATION_BWD
  CUDNN_ACTIVATION_FWD
  CUDNN_BATCHNORM_BWD
  CUDNN_BATCHNORM_FWD
  CUDNN_CONV_BIAS_ACTIVATION_FWD
  CUDNN_CONV_BWD_BIAS
  CUDNN_CONV_BWD_DATA
  CUDNN_CONV_BWD_FILTER
  CUDNN_CONV_FWD
  CUDNN_DROPOUT_BWD
  CUDNN_DROPOUT_FWD
  CUDNN_POOLING_BWD
  CUDNN_POOLING_FWD
  CUDNN_SOFTMAX_BWD
  CUDNN_SOFTMAX_FWD
)

for b in "${cudnn_bmarks[@]}"; do
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE ./bench --benchmark_filter="$b.*" --benchmark_out_format=json --benchmark_out=$OUT_DIR/`hostname`_"${b,,}".json
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE ./bench --benchmark_filter="$b.*" --benchmark_out_format=csv --benchmark_out=$OUT_DIR/`hostname`_"${b,,}".csv
done
