
#pragma once

#include <cuda_runtime.h>

#include "logger.hpp"

/**
 * \brief The corresponding error message is printed to \p stderr (or \p stdout
 * in device code) along with the supplied source context.
 *
 * \return The CUDA error.
 */
__host__ __device__ static inline cudaError_t
cuda_perror_impl(cudaError_t error, const char *filename, int line) {
  (void)filename;
  (void)line;
  if (error) {
#if !defined(__CUDA_ARCH__)
    LOG(critical, "CUDA error {} [{}, {}]: {}", error, filename, line,
        cudaGetErrorString(error));
#else
    printf("CUDA error %d [%s, %d]\n", error, filename, line);
#endif
  }
  return error;
}

/**
 * \brief Perror macro
 */
#ifndef CUDA_PERROR
#define CUDA_PERROR(e) cuda_perror_impl((cudaError_t)(e), __FILE__, __LINE__)
#endif

/**
 * \brief Perror macro with exit
 */
#ifndef CUDA_PERROR_EXIT
#define CUDA_PERROR_EXIT(e)                                                    \
  if (cuda_perror_impl((cudaError_t)(e), __FILE__, __LINE__)) {                \
    exit(1);                                                                   \
  }
#endif
