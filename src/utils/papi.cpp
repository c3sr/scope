
#include "b5_papi.hpp"
#include <papi.h>

/*
MIT License

Copyright (c) 2017 jpmag

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "matrix.hpp"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <random>

// https://developer.nvidia.com/nvidia-perfkit
// https://github.com/GPUOpen-Tools/GPA/blob/master/Doc/GPUPerfAPI-UserGuide.pdf

class PAPIFixture : public ::benchmark::Fixture {
public:
  void SetUp(benchmark::State&) override {
    printf("----------------------\n");
    papi.init({PAPI_L1_DCM, PAPI_L2_DCM, PAPI_L1_ICM, PAPI_L2_ICM, PAPI_FP_INS});
    papi.start();
  }
  void TearDown(benchmark::State& st) override {
    papi.read();
    papi.stop();
    papi.print();
    for (auto& evt : papi) {
      st.counters[papi.event_str(evt.first)] = evt.second;
    }
  }

  b5::PAPICounters papi;
};

BENCHMARK_DEFINE_F(PAPIFixture, MatAccessRoundBrackets)(benchmark::State& st) {
  int n = st.range(0);
  matrix<float> A(n, n, 1);
  float a = 0;
  while (st.KeepRunning()) {
    for (int i = 0; i < A.nrows; ++i) {
      for (int j = 0; j < A.ncols; ++j) {
        a += A[i][j];
      }
    }
  }
  printf("a=%f\n", a);
}
BENCHMARK_DEFINE_F(PAPIFixture, MatAccessSquareBrackets)(benchmark::State& st) {
  int n = st.range(0);
  matrix<float> A(n, n, 1);
  float a = 0;
  while (st.KeepRunning()) {
    for (int i = 0; i < A.nrows; ++i) {
      for (int j = 0; j < A.ncols; ++j) {
        a += A[i][j];
      }
    }
  }
  printf("a=%f\n", a);
}
BENCHMARK_DEFINE_F(PAPIFixture, MatAccessArrOperator)(benchmark::State& st) {
  int n = st.range(0);
  matrix<float> A(n, n, 1);
  float a = 0;
  while (st.KeepRunning()) {
    for (int i = 0; i < A.nrows; ++i) {
      float* row = A[i];
      for (int j = 0; j < A.ncols; ++j) {
        a += row[j];
      }
    }
  }
  printf("a=%f\n", a);
}
BENCHMARK_REGISTER_F(PAPIFixture, MatAccessRoundBrackets)->Arg((1 << 8) + 1);
BENCHMARK_REGISTER_F(PAPIFixture, MatAccessSquareBrackets)->Arg((1 << 8) + 1);
BENCHMARK_REGISTER_F(PAPIFixture, MatAccessArrOperator)->Arg((1 << 8) + 1);
BENCHMARK_REGISTER_F(PAPIFixture, MatAccessRoundBrackets)->Arg((1 << 10) + 1);
BENCHMARK_REGISTER_F(PAPIFixture, MatAccessSquareBrackets)->Arg((1 << 10) + 1);
BENCHMARK_REGISTER_F(PAPIFixture, MatAccessArrOperator)->Arg((1 << 10) + 1);
BENCHMARK_REGISTER_F(PAPIFixture, MatAccessRoundBrackets)->Arg((1 << 14) + 1);
BENCHMARK_REGISTER_F(PAPIFixture, MatAccessSquareBrackets)->Arg((1 << 14) + 1);
BENCHMARK_REGISTER_F(PAPIFixture, MatAccessArrOperator)->Arg((1 << 14) + 1);

template <typename T, class Func>
void create_and_call(benchmark::State& st, int SizeA, int SizeAB, int SizeB, Func fn) {
  matrix<T> A(SizeA, SizeAB, 1), B(SizeAB, SizeB, 2), C;
  while (st.KeepRunning()) {
    // benchmark::DoNotOptimize(C);
    fn(A, B, &C);
  }
}
template <typename T, class Func>
void create_and_call4(benchmark::State& st, int SizeA, int SizeAB, int SizeB, Func fn) {
  matrix<T> A(SizeA, SizeAB, 1), B(SizeAB, SizeB, 2), C, D;
  while (st.KeepRunning()) {
    // benchmark::DoNotOptimize(C);
    fn(A, B, &C, &D);
  }
}

BENCHMARK_DEFINE_F(PAPIFixture, FMatMultNaiveBad)(benchmark::State& st) {
  int n = st.range(0);
  create_and_call<float>(st, n, n, n, &matrix<float>::mult_naive_bad);
}
BENCHMARK_DEFINE_F(PAPIFixture, FMatMultNaive)(benchmark::State& st) {
  int n = st.range(0);
  create_and_call<float>(st, n, n, n, &matrix<float>::mult_naive);
}
BENCHMARK_DEFINE_F(PAPIFixture, FMatMultNaiveBetter)(benchmark::State& st) {
  int n = st.range(0);
  create_and_call<float>(st, n, n, n, &matrix<float>::mult_naive_better);
}
BENCHMARK_DEFINE_F(PAPIFixture, FMatMultNaiveTranspose)(benchmark::State& st) {
  int n = st.range(0);
  create_and_call4<float>(st, n, n, n, &matrix<float>::mult_naive_transposed);
}

BENCHMARK_DEFINE_F(PAPIFixture, DMatMultNaiveBad)(benchmark::State& st) {
  int n = st.range(0);
  create_and_call<double>(st, n, n, n, &matrix<double>::mult_naive_bad);
}
BENCHMARK_DEFINE_F(PAPIFixture, DMatMultNaive)(benchmark::State& st) {
  int n = st.range(0);
  create_and_call<double>(st, n, n, n, &matrix<double>::mult_naive);
}
BENCHMARK_DEFINE_F(PAPIFixture, DMatMultNaiveBetter)(benchmark::State& st) {
  int n = st.range(0);
  create_and_call<double>(st, n, n, n, &matrix<double>::mult_naive_better);
}
BENCHMARK_DEFINE_F(PAPIFixture, DMatMultNaiveTranspose)(benchmark::State& st) {
  int n = st.range(0);
  create_and_call4<double>(st, n, n, n, &matrix<double>::mult_naive_transposed);
}

BENCHMARK_REGISTER_F(PAPIFixture, FMatMultNaiveBad)->Arg((1 << 6) + 1);
BENCHMARK_REGISTER_F(PAPIFixture, FMatMultNaive)->Arg((1 << 6) + 1);
BENCHMARK_REGISTER_F(PAPIFixture, FMatMultNaiveBetter)->Arg((1 << 6) + 1);
BENCHMARK_REGISTER_F(PAPIFixture, FMatMultNaiveTranspose)->Arg((1 << 6) + 1);
BENCHMARK_REGISTER_F(PAPIFixture, FMatMultNaiveBad)->Arg((1 << 8) + 1);
BENCHMARK_REGISTER_F(PAPIFixture, FMatMultNaive)->Arg((1 << 8) + 1);
BENCHMARK_REGISTER_F(PAPIFixture, FMatMultNaiveBetter)->Arg((1 << 8) + 1);
BENCHMARK_REGISTER_F(PAPIFixture, FMatMultNaiveTranspose)->Arg((1 << 8) + 1);
BENCHMARK_REGISTER_F(PAPIFixture, FMatMultNaiveBad)->Arg((1 << 10) + 1)->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(PAPIFixture, FMatMultNaive)->Arg((1 << 10) + 1)->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(PAPIFixture, FMatMultNaiveBetter)->Arg((1 << 10) + 1)->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(PAPIFixture, FMatMultNaiveTranspose)->Arg((1 << 10) + 1)->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(PAPIFixture, FMatMultNaiveBad)->Arg((1 << 12) + 1)->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(PAPIFixture, FMatMultNaive)->Arg((1 << 12) + 1)->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(PAPIFixture, FMatMultNaiveBetter)->Arg((1 << 12) + 1)->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(PAPIFixture, FMatMultNaiveTranspose)->Arg((1 << 12) + 1)->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(PAPIFixture, DMatMultNaiveBad)->Arg((1 << 6) + 1);
BENCHMARK_REGISTER_F(PAPIFixture, DMatMultNaive)->Arg((1 << 6) + 1);
BENCHMARK_REGISTER_F(PAPIFixture, DMatMultNaiveBetter)->Arg((1 << 6) + 1);
BENCHMARK_REGISTER_F(PAPIFixture, DMatMultNaiveTranspose)->Arg((1 << 6) + 1);
BENCHMARK_REGISTER_F(PAPIFixture, DMatMultNaiveBad)->Arg((1 << 8) + 1);
BENCHMARK_REGISTER_F(PAPIFixture, DMatMultNaive)->Arg((1 << 8) + 1);
BENCHMARK_REGISTER_F(PAPIFixture, DMatMultNaiveBetter)->Arg((1 << 8) + 1);
BENCHMARK_REGISTER_F(PAPIFixture, DMatMultNaiveTranspose)->Arg((1 << 8) + 1);
BENCHMARK_REGISTER_F(PAPIFixture, DMatMultNaiveBad)->Arg((1 << 10) + 1)->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(PAPIFixture, DMatMultNaive)->Arg((1 << 10) + 1)->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(PAPIFixture, DMatMultNaiveBetter)->Arg((1 << 10) + 1)->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(PAPIFixture, DMatMultNaiveTranspose)->Arg((1 << 10) + 1)->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(PAPIFixture, DMatMultNaiveBad)->Arg((1 << 12) + 1)->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(PAPIFixture, DMatMultNaive)->Arg((1 << 12) + 1)->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(PAPIFixture, DMatMultNaiveBetter)->Arg((1 << 12) + 1)->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(PAPIFixture, DMatMultNaiveTranspose)->Arg((1 << 12) + 1)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN()
