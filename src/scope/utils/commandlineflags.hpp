#ifndef BENCHMARK_COMMANDLINEFLAGS_H_
#define BENCHMARK_COMMANDLINEFLAGS_H_

#include <cstdint>
#include <string>
#include <vector>

namespace bench {
namespace flags {}
} // namespace bench

// Macro for referencing flags.
#define FLAG(name) bench::flags::name

#define FLAGS_NS(stmt)                                                                                                 \
  namespace bench {                                                                                                    \
    namespace flags {                                                                                                  \
      stmt;                                                                                                            \
    }                                                                                                                  \
  }

// Macros for declaring flags.
#define DECLARE_FLAG_bool(name) FLAGS_NS(extern bool name)
#define DECLARE_FLAG_int32(name) FLAGS_NS(extern int32_t name)
#define DECLARE_FLAG_int64(name) FLAGS_NS(extern int64_t name)
#define DECLARE_FLAG_double(name) FLAGS_NS(extern double name)
#define DECLARE_FLAG_string(name) FLAGS_NS(extern std::string name)
#define DECLARE_FLAG_vec_int32(name) FLAGS_NS(extern std::vector<int32_t> name)

// Macros for defining flags.
#define DEFINE_FLAG_bool(name, default_val, doc) FLAGS_NS(bool name = (default_val))
#define DEFINE_FLAG_int32(name, default_val, doc) FLAGS_NS(int32_t name = (default_val))
#define DEFINE_FLAG_int64(name, default_val, doc) FLAGS_NS(int64_t name = (default_val))
#define DEFINE_FLAG_double(name, default_val, doc) FLAGS_NS(double name = (default_val))
#define DEFINE_FLAG_string(name, default_val, doc) FLAGS_NS(std::string name = (default_val))
#define DEFINE_FLAG_vec_int32(name, default_val, doc) FLAGS_NS(std::vector<int32_t> name(default_val))

#endif // BENCHMARK_COMMANDLINEFLAGS_H_
