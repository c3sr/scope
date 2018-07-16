#pragma once

#include <cudnn.h>

extern cudnnHandle_t cudnn_handle;

bool init_cudnn();
