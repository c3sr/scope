
#include <benchmark/benchmark.h>

#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <cuda_runtime.h>

#include "init/init.hpp"
#include "utils/utils.hpp"
#include "Tangram_axpy/args.hpp"

template <typename T>
__device__ void zaxpy_1D_53(T *input_X_54, T *input_Y_55,
                            T *output_Z_56, int ObjectSize_59,
                            int Stride_60, const T a_57, int tile_36) {

  //unsigned int tid_61 = threadIdx.x;
  for (int i_58 = 0; (i_58 < ObjectSize_59); i_58 += Stride_60) {
    if(i_58+threadIdx.x < tile_36)
        output_Z_56[i_58] = (input_Y_55[i_58] + (a_57 * input_X_54[i_58]));
  }
}

template <typename T>
__global__ void zaxpy_1D_41(T *input_X_42, int ObjectSize_62,
                            T *input_Y_43, int ObjectSize_63,
                            T *output_Z_44, int ObjectSize_64,
                            int ObjectSize_65, const T a_45, int tile_36) {

  //unsigned int blockID_66 = blockIdx.x;
  int p_46 = blockDim.x;
  int x_size_47 = ObjectSize_65;
  int tile_48 = ((((x_size_47 + p_46) - 1)) / p_46);

  T *part_z_52 = output_Z_44 + (blockIdx.x * ObjectSize_64);
  T *part_x_50 = input_X_42 + (blockIdx.x * ObjectSize_62);
  T *part_y_51 = input_Y_43 + (blockIdx.x * ObjectSize_63);
  /*Map*/
  zaxpy_1D_53<T>(part_x_50 + (0 + (threadIdx.x * 1)),
              part_y_51 + (0 + (threadIdx.x * 1)),
              part_z_52 + (0 + (threadIdx.x * 1)), (tile_48 * p_46),
              p_46, a_45, tile_36);
}

template <typename T, unsigned int TGM_TEMPLATE_0, unsigned int TGM_TEMPLATE_1>
void zaxpy_1D_29(T *input_X_30, T *input_Y_31, T *output_Z_32,
                 int ObjectSize_67, const T a_33, benchmark::State &state) {

  int p_34 = TGM_TEMPLATE_0;
  int x_size_35 = ObjectSize_67;
  int tile_36 = ((((x_size_35 + p_34) - 1)) / p_34);

  T *part_z_40 = output_Z_32;
  T *part_x_38 = input_X_30;
  T *part_y_39 = input_Y_31;
  /*Map*/
  dim3 dimBlock(TGM_TEMPLATE_1);
  dim3 dimGrid(p_34);

#ifdef USE_CUDA_EVENTS
  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));
#endif // USE_CUDA_EVENTS

  for (auto _ : state) {
#ifdef USE_CUDA_EVENTS
    cudaEventRecord(start, NULL);
#endif // USE_CUDA_EVENTS

    zaxpy_1D_41<T><<<dimGrid, dimBlock>>>(part_x_38, tile_36, part_y_39, tile_36,
                                     part_z_40, tile_36, tile_36, a_33, tile_36);
#ifdef USE_CUDA_EVENTS
    cudaEventRecord(stop, NULL);
    const auto cuda_err = cudaEventSynchronize(stop);
#else  // USE_CUDA_EVENTS
    const auto cuda_err = cudaDeviceSynchronize();
#endif // USE_CUDA_EVENTS

    state.PauseTiming();
    if (PRINT_IF_ERROR(cuda_err)) {
      state.SkipWithError("CUDA/VECTOR_ADD/BASIC failed to launch kernel");
      break;
    }
#ifdef USE_CUDA_EVENTS
    float msecTotal = 0.0f;
    if (PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop))) {
      state.SkipWithError("CUDA/VECTOR_ADD/BASIC failed to get elapsed time");
      break;
    }
    state.SetIterationTime(msecTotal / 1000);
#endif // USE_CUDA_EVENTS
    state.ResumeTiming();
  }

state.counters.insert({{"N", x_size_35},
                         {"GRID_SIZE", p_34},
                         {"BLOCK_SIZE", TGM_TEMPLATE_1},
                         {"Flops", {state.iterations() * 1.0 * x_size_35, benchmark::Counter::kAvgThreadsRate}}});
  state.SetBytesProcessed(int64_t(state.iterations()) * x_size_35 * sizeof(T));
  state.SetItemsProcessed(int64_t(state.iterations()) * x_size_35);

}

template <typename T, unsigned int TGM_TEMPLATE_0, unsigned int TGM_TEMPLATE_1>
void zaxpy_kernel_launcher_1(const T a_6,
                             const T *X_7, const T *Y_8,
                             T *Z_9, benchmark::State &state) {

  int iStart_25 = 0;
  int iEnd_26 = state.range(0);
  T *input_X_24;
  cudaMalloc((void **)&input_X_24, ((iEnd_26 - iStart_25)) * sizeof(T));
  cudaMemcpy(input_X_24, X_7 + iStart_25,
             ((iEnd_26 - iStart_25)) * sizeof(T), cudaMemcpyHostToDevice);
  T *input_Y_27;
  cudaMalloc((void **)&input_Y_27, ((iEnd_26 - iStart_25)) * sizeof(T));
  cudaMemcpy(input_Y_27, Y_8 + iStart_25,
             ((iEnd_26 - iStart_25)) * sizeof(T), cudaMemcpyHostToDevice);
  T *output_Z_28;
  cudaMalloc((void **)&output_Z_28, ((iEnd_26 - iStart_25)) * sizeof(T));
  cudaMemcpy(output_Z_28, Z_9 + iStart_25,
             ((iEnd_26 - iStart_25)) * sizeof(T), cudaMemcpyHostToDevice);

  zaxpy_1D_29<T, TGM_TEMPLATE_0, TGM_TEMPLATE_1>(
      input_X_24, input_Y_27, output_Z_28, (iEnd_26 - iStart_25), a_6, state);

  cudaMemcpy(Z_9 + iStart_25, output_Z_28,
             ((iEnd_26 - iStart_25)) * sizeof(T), cudaMemcpyDeviceToHost);

  cudaFree(input_X_24);
  cudaFree(input_Y_27);
  cudaFree(output_Z_28);

}

template <typename T, unsigned int TGM_TEMPLATE_0 = 32, unsigned int TGM_TEMPLATE_1 = 32>
static void TANGRAM_CUDA_AXPY(benchmark::State &state) {
  if (!has_cuda) {
    state.SkipWithError("CUDA/VECTOR_ADD/BASIC no CUDA device found");
    return;
  }

  const size_t N = state.range(0);

  auto x = std::vector<T>(N);
  auto y = std::vector<T>(N);
  auto z = std::vector<T>(N);

  std::fill(x.begin(), x.end(), 1);
  std::fill(y.begin(), y.end(), 1);
  std::fill(z.begin(), z.end(), 0);

  T a = 1;

  zaxpy_kernel_launcher_1<T, TGM_TEMPLATE_0, TGM_TEMPLATE_1>(a, x.data(), y.data(), z.data(), state);

}

#ifdef USE_CUDA_EVENTS
#ifndef FAST_MODE
//BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, char, 1, 32)->ALL_ARGS()->UseManualTime();
//BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, int, 1, 32)->ALL_ARGS()->UseManualTime();
//BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, float, 1, 32)->ALL_ARGS()->UseManualTime();

BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 1, 32)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 2, 32)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 4, 32)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 8, 32)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 16, 32)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 32, 32)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 64, 32)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 128, 32)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 256, 32)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 512, 32)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 1024, 32)->ALL_ARGS()->UseManualTime();

BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 1, 64)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 2, 64)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 4, 64)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 8, 64)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 16, 64)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 32, 64)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 64, 64)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 128, 64)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 256, 64)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 512, 64)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 1024, 64)->ALL_ARGS()->UseManualTime();

BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 1, 128)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 2, 128)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 4, 128)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 8, 128)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 16, 128)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 32, 128)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 64, 128)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 128, 128)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 256, 128)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 512, 128)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 1024, 128)->ALL_ARGS()->UseManualTime();

BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 1, 256)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 2, 256)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 4, 256)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 8, 256)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 16, 256)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 32, 256)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 64, 256)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 128, 256)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 256, 256)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 512, 256)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 1024, 256)->ALL_ARGS()->UseManualTime();

BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 1, 512)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 2, 512)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 4, 512)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 8, 512)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 16, 512)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 32, 512)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 64, 512)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 128, 512)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 256, 512)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 512, 512)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 1024, 512)->ALL_ARGS()->UseManualTime();

BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 1, 1024)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 2, 1024)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 4, 1024)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 8, 1024)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 16, 1024)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 32, 1024)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 64, 1024)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 128, 1024)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 256, 1024)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 512, 1024)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 1024, 1024)->ALL_ARGS()->UseManualTime();

#if 0
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, char, 1, 64)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, int, 1, 64)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, float, 1, 64)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 1, 64)->ALL_ARGS()->UseManualTime();

BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, char, 1, 128)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, int, 1, 128)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, float, 1, 128)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 1, 128)->ALL_ARGS()->UseManualTime();

BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, char, 1, 256)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, int, 1, 256)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, float, 1, 256)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 1, 256)->ALL_ARGS()->UseManualTime();
#endif
#endif // FAST_MODE

#if 0
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, char, 1, 512)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, int, 1, 512)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, float, 1, 512)->ALL_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 1, 512)->ALL_ARGS()->UseManualTime();
#endif
#else // USE_CUDA_EVENTS
#ifndef FAST_MODE
#if 0
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, char, 1, 32)->ALL_ARGS();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, int, 1, 32)->ALL_ARGS();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, float, 1, 32)->ALL_ARGS();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 1, 32)->ALL_ARGS();

BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, char, 1, 64)->ALL_ARGS();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, int, 1, 64)->ALL_ARGS();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, float, 1, 64)->ALL_ARGS();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 1, 64)->ALL_ARGS();

BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, char, 1, 128)->ALL_ARGS();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, int, 1, 128)->ALL_ARGS();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, float, 1, 128)->ALL_ARGS();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 1, 128)->ALL_ARGS();

BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, char, 1, 256)->ALL_ARGS();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, int, 1, 256)->ALL_ARGS();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, float, 1, 256)->ALL_ARGS();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 1, 256)->ALL_ARGS();
#endif
#endif // FAST_MODE
#if 0
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, char, 1, 512)->ALL_ARGS();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, int, 1, 512)->ALL_ARGS();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, float, 1, 512)->ALL_ARGS();
BENCHMARK_TEMPLATE(TANGRAM_CUDA_AXPY, double, 1, 512)->ALL_ARGS();
#endif
#endif // USE_CUDA_EVENTS
