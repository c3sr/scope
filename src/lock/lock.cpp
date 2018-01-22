#include <benchmark/benchmark.h>

#include <pthread.h>
#include <stdio.h>

#include "init/init.hpp"
#include "utils/utils.hpp"

#include "lock/args.hpp"

static std::string mutex_type_string(const int type) {
  switch (type) {
    case PTHREAD_MUTEX_NORMAL:
      return "PTHREAD_MUTEX_NORMAL";
    case PTHREAD_MUTEX_ERRORCHECK:
      return "PTHREAD_MUTEX_ERRORCHECK";
    case PTHREAD_MUTEX_RECURSIVE:
      return "PTHREAD_MUTEX_RECURSIVE";
    default:
      return "UNDEFINED";
  }
}

static void lock_impl(int iter, int mutex_type) {
  pthread_mutex_t my_mutex;
  pthread_mutexattr_t my_attr;
  pthread_mutexattr_init(&my_attr);
  pthread_mutexattr_settype(&my_attr, mutex_type); // Make the mutex reenterant
  pthread_mutex_init(&my_mutex, &my_attr);

  int i = 0;
  while (i < iter) { // Only one thread locks and unlocks the lock
    pthread_mutex_lock(&my_mutex);

    benchmark::DoNotOptimize(i);

    pthread_mutex_unlock(&my_mutex);
    i++;
  }
}

template <int mutex_type>
static void LOCK(benchmark::State& state) {
  static const std::string mutex_type_name = mutex_type_string(mutex_type);
  state.SetLabel(fmt::format("LOCK/{}", mutex_type_name));

  const auto N = state.range(0);

  for (auto _ : state) {
    lock_impl(N, mutex_type);
    benchmark::DoNotOptimize(N);
  }

  state.counters.insert({{"N", N}});
  state.SetItemsProcessed(int64_t(state.iterations()) * N);
}

BENCHMARK_TEMPLATE(LOCK, PTHREAD_MUTEX_NORMAL)->ARGS_FULL();
BENCHMARK_TEMPLATE(LOCK, PTHREAD_MUTEX_ERRORCHECK)->ARGS_FULL();
BENCHMARK_TEMPLATE(LOCK, PTHREAD_MUTEX_RECURSIVE)->ARGS_FULL();
