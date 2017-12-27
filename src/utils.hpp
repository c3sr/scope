#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#pragma once

#include <type_traits>
#include <utility>

template <class Function> class defer {
public:
  template <class F>
  explicit defer(F &&f) noexcept : defer_function_(std::forward<F>(f)) {}

  ~defer() { defer_function_(); }

private:
  Function defer_function_;
};

template <class F>
defer<typename std::decay<F>::type> make_defer(F &&defer_function) {
  return defer<typename std::decay<F>::type>(std::forward<F>(defer_function));
}

#endif // __UTILS_HPP__
