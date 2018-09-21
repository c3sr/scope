#pragma once

#include <unistd.h>

static ALWAYS_INLINE size_t page_size() {
    return sysconf(_SC_PAGESIZE);
}