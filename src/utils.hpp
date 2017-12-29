#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#pragma once

#include <type_traits>
#include <utility>

#ifdef __GNUC__
#define UNUSED __attribute__((unused))
#else // __GNUC__
#define UNUSED
#endif // __GNUC__

template <class Function> class defer_func {
public:
  template <class F>
  explicit defer_func(F &&f) noexcept : defer_function_(std::forward<F>(f)) {}

  ~defer_func() { defer_function_(); }

private:
  Function defer_function_;
};

template <class F>
defer_func<typename std::decay<F>::type> make_defer(F &&defer_function) {
  return defer_func<typename std::decay<F>::type>(
      std::forward<F>(defer_function));
}

#define DEFER_1(x, y) x##y
#define DEFER_2(x, y) DEFER_1(x, y)
#define DEFER_3(x) DEFER_2(x, __COUNTER__)
#define defer(code) auto DEFER_3(_defer_) = make_defer([&]() { code; })

#endif // __UTILS_HPP__
