

#pragma once

#ifdef __GNUC__
#define UNUSED __attribute__((unused))
#else // __GNUC__
#define UNUSED
#endif // __GNUC__