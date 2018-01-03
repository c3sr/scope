#include <benchmark/benchmark.h>

#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <cblas.h>

#include "utils/utils.hpp"

#include "axpy/args.hpp"
#include "axpy/utils.hpp"

template <typename T>
static void CBLAS(benchmark::State &state) {

  static const std::string IMPLEMENTATION_NAME = axpy::detail::implementation_name<T>();
  state.SetLabel(fmt::format("CBLAS/{}", IMPLEMENTATION_NAME));

  const T one{1};

  const auto N      = state.range(0);
  const auto x_incr = state.range(1);
  const auto y_incr = state.range(2);

  auto x = std::vector<T>(N);
  auto y = std::vector<T>(N);
  T alpha{0.5};

  std::fill(x.begin(), x.end(), one);
  std::fill(y.begin(), y.end(), one);

  for (auto _ : state) {

    if constexpr (std::is_same<T, float>::value) {
      cblas_saxpy(N, alpha, x.data(), x_incr, y.data(), y_incr);
    } else if constexpr (std::is_same<T, double>::value) {
      cblas_daxpy(N, alpha, x.data(), x_incr, y.data(), y_incr);
    } else if constexpr (std::is_same<T, std::complex<float>>::value) {
      using scalar_type = typename T::value_type;
      cblas_caxpy(N, reinterpret_cast<scalar_type(&)[2]>(alpha), reinterpret_cast<scalar_type *>(x.data()), x_incr,
                  reinterpret_cast<scalar_type *>(y.data()), y_incr);
    } else if constexpr (std::is_same<T, std::complex<double>>::value) {
      using scalar_type = typename T::value_type;
      cblas_zaxpy(N, reinterpret_cast<scalar_type(&)[2]>(alpha), reinterpret_cast<scalar_type *>(x.data()), x_incr,
                  reinterpret_cast<scalar_type *>(y.data()), y_incr);
    }
  }

  state.counters.insert({{"N", N},
                         {"x_increment", x_incr},
                         {"y_increment", y_incr},
                         {"Flops", {3.0 * N, benchmark::Counter::kAvgThreadsRate}}});

  state.SetBytesProcessed(int64_t(state.iterations()) * 3 * N * sizeof(T));
  state.SetItemsProcessed(int64_t(state.iterations()) * 3 * N);
}

static void CBLAS_SAXPY(benchmark::State &state) {
  CBLAS<float>(state);
}

static void CBLAS_DAXPY(benchmark::State &state) {
  CBLAS<double>(state);
}

static void CBLAS_CAXPY(benchmark::State &state) {
  CBLAS<std::complex<float>>(state);
}

static void CBLAS_ZAXPY(benchmark::State &state) {
  CBLAS<std::complex<double>>(state);
}

BENCHMARK(CBLAS_DAXPY)->ARGS_FULL();
