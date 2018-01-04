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
void cblas_axpy(const int N, const T alpha, const T* X, const int x_incr, T* Y, const int y_incr);

template <>
void cblas_axpy<float>(const int N, const float alpha, const float* X, const int x_incr, float* Y, const int y_incr) {

  cblas_saxpy(N, alpha, X, x_incr, Y, y_incr);
}

template <>
void cblas_axpy<double>(const int N, const double alpha, const double* X, const int x_incr, double* Y,
                        const int y_incr) {

  cblas_daxpy(N, alpha, X, x_incr, Y, y_incr);
}

template <>
void cblas_axpy<std::complex<float>>(const int N, const std::complex<float> alpha, const std::complex<float>* X,
                                     const int x_incr, std::complex<float>* Y, const int y_incr) {

  using scalar_type = float;

  cblas_caxpy(N, reinterpret_cast<const scalar_type(&)[2]>(alpha), reinterpret_cast<const scalar_type*>(X), x_incr,
              reinterpret_cast<scalar_type*>(Y), y_incr);
}

template <>
void cblas_axpy<std::complex<double>>(const int N, const std::complex<double> alpha, const std::complex<double>* X,
                                      const int x_incr, std::complex<double>* Y, const int y_incr) {

  using scalar_type = double;

  cblas_zaxpy(N, reinterpret_cast<const scalar_type(&)[2]>(alpha), reinterpret_cast<const scalar_type*>(X), x_incr,
              reinterpret_cast<scalar_type*>(Y), y_incr);
}

template <typename T>
static void CBLAS(benchmark::State& state) {

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
    cblas_axpy<T>(N, alpha, x.data(), x_incr, y.data(), y_incr);
  }

  state.counters.insert({{"N", N},
                         {"x_increment", x_incr},
                         {"y_increment", y_incr},
                         {"Flops", {3.0 * N, benchmark::Counter::kAvgThreadsRate}}});

  state.SetBytesProcessed(int64_t(state.iterations()) * 3 * N * sizeof(T));
  state.SetItemsProcessed(int64_t(state.iterations()) * 3 * N);
}

static void CBLAS_SAXPY(benchmark::State& state) {
  CBLAS<float>(state);
}

static void CBLAS_DAXPY(benchmark::State& state) {
  CBLAS<double>(state);
}

static void CBLAS_CAXPY(benchmark::State& state) {
  CBLAS<std::complex<float>>(state);
}

static void CBLAS_ZAXPY(benchmark::State& state) {
  CBLAS<std::complex<double>>(state);
}

BENCHMARK(CBLAS_DAXPY)->ARGS_FULL();
