#! /bin/bash

# Run all NUMA benchmarks in their own process

set -eou pipefail -x

# default out dir is current directory
OUT_DIR=`readlink -f .`

# default bench command is scope
BENCH=./scope


while getopts "h?o:b:" opt; do
    case "$opt" in
    h|\?)
        show_help
        exit 0
        ;;
    o)  OUT_DIR=`readlink -f $OPTARG`
        ;;
    b)  BENCH="$OPTARG";
        ;;
    esac
done

shift $((OPTIND-1))

[ "${1:-}" = "--" ] && shift

# echo "Leftovers: $@"

s822lc_gpus=( 0 1 2 3 )
s822lc_numas=( 0 1 )
 
ac922_gpus=( 0 1 2 3 )
ac922_numas=( 0 8 )

dgx_gpus=( 0 1 2 3 4 5 6 7 )
dgx_numas=( 0 1 )

shared_bmarks=(
noop
)

numa_numa_bmarks=(
NUMA_Memcpy_HostToPinned
NUMA_WR
NUMA_RD
noop
)

numa_gpu_gpu_bmarks=(
NUMA_Memcpy_GPUToGPU
noop
)

gpu_gpu_bmarks=(
CUDA_Memcpy_GPUToGPU
UM_Coherence_GPUToGPU
UM_Prefetch_GPUToGPU
UM_Latency_GPUToGPU
noop
)

numa_gpu_bmarks=(
Comm_NUMAMemcpy_GPUToHost
Comm_NUMAMemcpy_GPUToPinned
Comm_NUMAMemcpy_HostToGPU
Comm_NUMAMemcpy_PinnedToGPU
Comm_UM_Coherence_GPUToHost
Comm_UM_Coherence_HostToGPU
Comm_UM_Prefetch_GPUToHost
Comm_UM_Prefetch_HostToGPU
Comm_UM_Latency_GPUToHost
Comm_UM_Latency_HostToGPU
noop
)


mkdir -p "$OUT_DIR"



for b in "${numa_gpu_bmarks[@]}"; do
    if [ "$b" != "noop" ]; then
        eval "$BENCH" --numa_ids=0 --cuda_ids=0 --benchmark_filter="$b/.*" --benchmark_out="$OUT_DIR"/`hostname`-$b-numa0-cuda0.json --benchmark_repetitions=5;
        eval "$BENCH" --numa_ids=0 --cuda_ids=1 --benchmark_filter="$b/.*" --benchmark_out="$OUT_DIR"/`hostname`-$b-numa0-cuda1.json --benchmark_repetitions=5;
    fi
done
