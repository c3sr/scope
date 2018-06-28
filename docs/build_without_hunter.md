# Building Without Hunter

Instead of allowing Hunter to download the prerequisites, you can download and provide them yourself.
Microbench would then be configured in the following way

    cmake .. \
        -DCONFIG_USE_HUNTER=OFF \
        -DSUGAR_ROOT=${SUGAR_ROOT}

`-DCONFIG_USE_HUNTER=OFF` disables hunter. In that case, you need to provide the software provided on this page.
`-DSUGAR_ROOT` tells CMake where to find Sugar, which microbench uses to find source files.
If you install any of the libraries in an unusual place, you will have to use `-DCMAKE_PREFIX_PATH`, a `;`-delimited list of install locations for the libraries.
If you install a library with the prefix `<path>`, you probably want to add `<path>` to `-DCMAKE_PREFIX_PATH`.

    cmake .. \
        -DCONFIG_USE_HUNTER=OFF \
        -DCMAKE_PREFIX_PATH="${BENCHMARK_ROOT};${CUB_ROOT};${FMT_ROOT};${SPDLOG_ROOT};${BENCHMARK_ROOT}" \
        -DSUGAR_ROOT=${SUGAR_ROOT}


## Sugar 1.3.0

[Github Release](https://github.com/ruslo/sugar/releases/tag/v1.3.0)

## Benchmark 1.4.1

[Github Release](https://github.com/google/benchmark/releases/tag/v1.4.1)

## CUB 1.8.0

[Github Release](https://github.com/NVlabs/cub/releases/tag/v1.8.0)

## fmt 4.0.0

spdlog uses fmt 4.0.0, so microbench does too.
[Github Release](https://github.com/fmtlib/fmt/releases/tag/4.0.0).

## spdlog 0.17.0

    apt-get install libspdlog-dev
    
or

[Github Release](https://github.com/gabime/spdlog/releases/tag/v0.17.0)