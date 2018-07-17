

#include <cudnn.h>

#include "scope/utils/utils.hpp"

cudnnHandle_t cudnn_handle;

bool init_cudnn() {
  return PRINT_IF_ERROR(cudnnCreate(&cudnn_handle));
}
