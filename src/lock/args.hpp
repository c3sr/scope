#pragma once

#define ARGS_FULL()                                                                                              \
  Args({1})                                                                                                      \
      ->Args({10})                                                                                               \
      ->Args({100})                                                                                              \
      ->Args({1000})                                                                                             \
      ->Args({10000})                                                                                            \
      ->Args({100000})                                                                                           \
      ->Args({10000000})                                                                                         \
      ->Args({100000000})                                                                                        \
      ->Args({1000000000})                                                                                       \
      ->ArgNames({"N"})
