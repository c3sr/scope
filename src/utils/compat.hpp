#pragma once


#ifdef __GNUC__
#define UNUSED __attribute__((unused))
#else // __GNUC__
#define UNUSED
#endif // __GNUC__

#if defined(__GNUC__)
#define ALWAYS_INLINE __attribute__((always_inline))
#else // defined(__GNUC__)
#define ALWAYS_INLINE __forceinline
#endif // defined(__GNUC__)