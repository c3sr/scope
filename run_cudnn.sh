#! /bin/bash

make -j && ./bench --benchmark_filter=CONV_FORWARD --benchmark_out_format=json --benchmark_out=../results/cudnn/volta_conv_fwd.json

