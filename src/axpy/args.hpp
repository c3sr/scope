#pragma once

#define ARGS_FULL()                                                                                                    \
  Args({10, 1, 1})                                                                                                     \
      ->Args({100, 1, 1})                                                                                              \
      ->Args({1000, 1, 1})                                                                                             \
      ->Args({10000, 1, 1})                                                                                            \
      ->Args({100000, 1, 1})                                                                                           \
      ->Args({10000000, 1, 1})                                                                                         \
      ->Args({100000000, 1, 1})                                                                                        \
      ->Args({1000000000, 1, 1})                                                                                       \
      ->ArgNames({"N", "INCX", "INCY"})
