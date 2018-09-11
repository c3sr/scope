# Scope

|master|
|--|
| [![Build Status](https://travis-ci.com/c3sr/scope.svg?branch=master)](https://travis-ci.com/c3sr/scope)|

A benchmark framework developed by the [IMPACT](impact.crhc.illinois.edu) group at the University of Illinois in collaboration with IBM through the the Center for Cognitive Computing Systems Research (C3SR).

Primary maintainers:
* Carl Pearson ([web](cwpearson.github.io)) ([email](mailto:pearson@illinois.edu))
* Abdul Dakkak ([email](mailto:dakkak@illinois.edu))
* Cheng Li ([email](mailto:cli99@illinois.edu))

Various benchmark suites using Scope are under development:

* [Comm|Scope](https://github.com/c3sr/comm_scope) - CUDA/NUMA data transfer performance (Carl Pearson, UIUC)
* [NCCL|Scope](https://github.com/rai-project/nccl_scope) - GPU collective communication performance (Sarah Hashash, UIUC)
* Histo|Scope - CUDA histogram techniques (Carl Pearson, UIUC)
* DDL|Scope - IBM Distributed Deep Learning Library benchmarks (Vandana Kulkarni, UIUC)
* [Misc|Scope](https://github.com/c3sr/misc_scope) - experimental or miscellaneous benchmarks


## Install CMake >= 3.12

On **x86-64**, the following may work.

```
cd /tmp
wget https://cmake.org/files/v3.12/cmake-3.12.0-Linux-x86_64.sh
sudo sh cmake-3.12.0-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir
```

On **ppc64le**, you will need to build from source.

## Compile

To compile the project run the following commands

    mkdir -p build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make
    
The build system uses Hunter to download all dependencies.
If you have trouble downloading dependencies, [check to make sure](docs/hunter_problems.md) Hunter/CMake can use SSL.
Or you can forego Hunter entirely and [provide your own dependencies](docs/build_without_hunter.md).

if you get errors about nvcc not supporting your gcc compiler, then you may want to use

    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_HOST_COMPILER=`which gcc-6` .. 

You can optionally choose your own CUDA archs that you would like to be compiled:

    cmake -DNVCC_ARCH_FLAGS="2.0 2.1 2.0 2.1 3.0 3.2 3.5 3.7 5.0 5.2 5.3" ..

The accepted syntax is the same as the `CUDA_SELECT_NVCC_ARCH_FLAGS` syntax in the FindCUDA module.

You can disable or enable individual scopes

    cmake -DENABLE_MISC=0 ...

The submodules should automatically be checked out.
If not, try checking them out yourself:

```
git submodule update --init --recursive
```

or to update modules to the proper verions

```
git submodule update --recursive --remote
```

## Available Benchmarks

The available benchmarks and descriptions are listed [here](docs/benchmark_descriptions.md). You can list all the benchmarks with

    ./scope --benchmark_list_tests=true

you can filter the benchmarks that are run with a regular expression passed to `--benchmark_filter`.

    ./scope --benchmark_filter=[regex]

for example

    ./scope --benchmark_filter=SGEMM

futher controls over the benchmarks are explained in the `--help` option

## Run all the benchmarks

This is not generally recommended, as it will take quite some time.

    ./scope

The above will output to stdout something like 

    ------------------------------------------------------------------------------
    Benchmark                       Time           CPU Iterations UserCounters...
    ------------------------------------------------------------------------------
    SGEMM/1000/1/1/-1/1             5 us          5 us     126475 K=1 M=1000 N=1 alpha=-1 beta=1
    SGEMM/128/169/1728/1/0        539 us        534 us       1314 K=1.728k M=128 N=169 alpha=1 beta=0
    SGEMM/128/729/1200/1/0       1042 us       1035 us        689 K=1.2k M=128 N=729 alpha=1 beta=0
    SGEMM/192/169/1728/1/0        729 us        724 us        869 K=1.728k M=192 N=169 alpha=1 beta=0
    SGEMM/256/169/1/1/1             9 us          9 us      75928 K=1 M=256 N=169 alpha=1 beta=1
    SGEMM/256/729/1/1/1            35 us         35 us      20285 K=1 M=256 N=729 alpha=1 beta=1
    SGEMM/384/169/1/1/1            18 us         18 us      45886 K=1 M=384 N=169 alpha=1 beta=1
    SGEMM/384/169/2304/1/0       2475 us       2412 us        327 K=2.304k M=384 N=169 alpha=1 beta=0
    SGEMM/50/1000/1/1/1            10 us         10 us      73312 K=1 M=50 N=1000 alpha=1 beta=1
    SGEMM/50/1000/4096/1/0       6364 us       5803 us        100 K=4.096k M=50 N=1000 alpha=1 beta=0
    SGEMM/50/4096/1/1/1            46 us         45 us      13491 K=1 M=50 N=4.096k alpha=1 beta=1
    SGEMM/50/4096/4096/1/0      29223 us      26913 us         20 K=4.096k M=50 N=4.096k alpha=1 beta=0
    SGEMM/50/4096/9216/1/0      55410 us      55181 us         10 K=9.216k M=50 N=4.096k alpha=1 beta=0
    SGEMM/96/3025/1/1/1            55 us         51 us      14408 K=1 M=96 N=3.025k alpha=1 beta=1
    SGEMM/96/3025/363/1/0        1313 us       1295 us        570 K=363 M=96 N=3.025k alpha=1 beta=0

Output as JSON using

    ./scope --benchmark_out_format=json --benchmark_out=test.json
    
or preferably 

    ./scope --benchmark_out_format=json --benchmark_out=`hostname`.json

Repeat benchmark runs with

    ./scope --benchmark_repetitions=5

## Plot Benchmark JSON files

Try the [ScopePlot](https://github.com/rai-project/scope_plot) python package.

    pip install scope_plot

## On Minsky With PowerAI

```
cd build && rm -fr * && OpenBLAS=/opt/DL/openblas cmake -DCMAKE_BUILD_TYPE=Release .. -DOpenBLAS=/opt/DL/openblas
```

## Disable CPU frequency scaling

If you see this error:

```
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
```

you might want to disable the CPU frequency scaling while running the benchmark:

```bash
sudo cpupower frequency-set --governor performance
./mybench
sudo cpupower frequency-set --governor powersave
```

## Run with Docker

Install `nvidia-docker`, then, list the available benchmarks.

    nvidia-docker run  --rm raiproject/microbench:amd64-latest bench --benchmark_list_tests

You can run benchmarks in the following way (probably with the `--benchmark_filter` flag).

    nvidia-docker run --privileged --rm -v `readlink -f .`:/data -u `id -u`:`id -g` raiproject/microbench:amd64-latest ./numa-separate-process.sh dgx bench /data/sync2

* `--privileged` is needed to set the NUMA policy if NUMA benchmarks are to be run.
* `` -v `readlink -f .`:/data `` maps the current directory into the container as `/data`.
* `` --benchmark_out=/data/\`hostname`.json `` tells the `bench` binary to write the json output files to `/data` in the container, which is mapped to the current directory.
* `` -u `id -u`:`id -g` `` tells docker to run as user `id -u` and group `id -g`, which is the current user and group. This means that files that docker produces will be modifiable from the host system without root permission.

## Hunter Toolchain File

If some of the third-party code compiled by hunter needs a different compiler, you can create a cmake toolchain file to set various cmake variables that will be globally used when building that code. You can then pass this file into cmake

    cmake -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake ...

## Adding a new benchmark

If you would like to develop a benchmark suite, read [here](docs/new_scope.md) for more information.
Also, check out the [Example|Scope](https://github.com/c3sr/example_scope) for a template to get started

## Third-Party Resources

* https://github.com/google/benchmark
