# Tools

## ScopePlot

## Generate Sugar Files

## Generate Runscripts

On some systems, it may be desirable to run each benchmark in a separate process.

`tools/generate_runscripts.py` is a Python3 script for generating a bash script to run any subset of benchmarks.
By default, each benchmark will produce a single output file, which is the benchmark name following the hostname of the system the benchmark is executed on.

```bash
# generate script to run benchmarks
tools/generate_runscripts.py > run_all.sh
# run benchmarks
./run_all.sh
```

To generate a script to run only some benchmarks:

```bash
tools/generate_runscripts.py --benchmark-filter="Comm_UM_Prefetch_Host.*(26|28)"
```

produces

```bash
#! /bin/bash
set -xeuo pipefail

/home/pearson/repos/scope/tools/../build/scope --benchmark_filter="Comm_UM_Prefetch_HostToGPU/log2\(N\):26/manual_time" --benchmark_out="kubuntu1804_Comm_UM_Prefetch_HostToGPU_log2(N)_26_manual_time.json"
/home/pearson/repos/scope/tools/../build/scope --benchmark_filter="Comm_UM_Prefetch_HostToGPU/log2\(N\):28/manual_time" --benchmark_out="kubuntu1804_Comm_UM_Prefetch_HostToGPU_log2(N)_28_manual_time.json"
```

To seeother options:

```bash
tools/generate_runscripts.py -h
```

```
usage: generate_runscripts.py [-h] [--benchmark-filter BENCHMARK_FILTER]
                              [--scope-path SCOPE_PATH]
                              [--benchmark-repetitions BENCHMARK_REPETITIONS]
                              [--no-use-hostname]

Generate script to run each benchmark in a new process.

optional arguments:
  -h, --help            show this help message and exit
  --benchmark-filter BENCHMARK_FILTER
                        passed to SCOPE through --benchmark_filter=
  --scope-path SCOPE_PATH
                        path to scope
  --benchmark-repetitions BENCHMARK_REPETITIONS
                        passed to SCOPE through --benchmark_repetitions=
  --no-use-hostname     don't prefix output with hostname
```

Additionally, any arguments passed after the `--` flag will be directly appended to each `scope` command.