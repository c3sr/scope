#! /bin/bash

# Run all NUMA benchmarks in their own process

set -eou pipefail -x

# default out dir is current directory
OUT_DIR=`readlink -f .`
DOCKER=true

# Look for nvidia-docker, otherwise use docker run --runtime=nvidia
if ! [ -x "$(command -v nvidia-docker)" ]; then
  echo 'Error: nvidia-docker is not installed.' >&2
  DOCKER_RUN="docker run --runtime=nvidia"
else
  DOCKER_RUN="nvidia-docker run"
fi


# default bench command is through docker
BENCH="$DOCKER_RUN --privileged --rm -v "$OUT_DIR":/data -u `id -u`:`id -g` raiproject/microbench:amd64-develop bench"

while getopts "h?o:b:" opt; do
    case "$opt" in
    h|\?)
        show_help
        exit 0
        ;;
    o)  OUT_DIR=`readlink -f $OPTARG`
        ;;
    b)  BENCH="$OPTARG";
        DOCKER=false;
        ;;
    esac
done

if [ $DOCKER ]; then
  BENCHMARK_OUT="--benchmark_out=/data"
else
  BENCHMARK_OUT="--benchmark_out=$OUT_DIR"
fi

shift $((OPTIND-1))

[ "${1:-}" = "--" ] && shift

# echo "Leftovers: $@"

# Which machine to use (s822lc, ac922, or dgx)
machine=$1


s822lc_gpus=( 0 1 2 3 )
s822lc_numas=( 0 1 )
 
ac922_gpus=( 0 1 2 3 )
ac922_numas=( 0 8 )

dgx_gpus=( 0 1 2 3 4 5 6 7 )
dgx_numas=( 0 1 )

numas=$machine\_numas[@]
gpus=$machine\_gpus[@]

shared_bmarks=(
noop
)

numa_numa_bmarks=(
NUMA_Memcpy_HostToPinned
#NUMA_WR
#NUMA_RD
noop
)

numa_gpu_gpu_bmarks=(
#NUMA_Memcpy_GPUToGPU
noop
)

gpu_gpu_bmarks=(
#CUDA_Memcpy_GPUToGPU
#UM_Coherence_GPUToGPU
#UM_Prefetch_GPUToGPU
noop
)

numa_gpu_bmarks=(
#NUMA_Memcpy_GPUToHost
#NUMA_Memcpy_GPUToPinned
#NUMA_Memcpy_GPUToWC
#NUMA_Memcpy_HostToGPU
#NUMA_Memcpy_PinnedToGPU
#NUMA_Memcpy_WCToGPU
#NUMAUM_Coherence_GPUToHost
#NUMAUM_Coherence_HostToGPU
#NUMAUM_Latency_GPUToHost
#NUMAUM_Latency_HostToGPU
#NUMAUM_Prefetch_GPUToHost
#NUMAUM_Prefetch_HostToGPU
noop
)


mkdir -p "$OUT_DIR"



regex="a^"
for b in "${shared_bmarks[@]}"; do
    if [ "$b" != "noop" ]; then
	regex=`echo -n "$b|$regex"`
    fi
done
eval "$BENCH" --benchmark_filter="$regex" "$BENCHMARK_OUT"/`hostname`-shared.json --benchmark_repetitions=5;

for b in "${numa_numa_bmarks[@]}"; do
    if [ "$b" != "noop" ]; then
        for n1 in ${!numas}; do
            for n2 in ${!numas}; do
                eval "$BENCH" --benchmark_filter="$b/.*/$n1/$n2/" "$BENCHMARK_OUT"/`hostname`-$b-$n1-$n2.json --benchmark_repetitions=5;
            done
        done
    fi
done

for b in "${numa_gpu_bmarks[@]}"; do
    if [ "$b" != "noop" ]; then
        for n in ${!numas}; do
            for g in ${!gpus}; do
                eval "$BENCH" --benchmark_filter="$b/.*/$n/$g/" "$BENCHMARK_OUT"/`hostname`-$b-$n-$g.json --benchmark_repetitions=5;
            done
        done
    fi
done

for b in "${gpu_gpu_bmarks[@]}"; do
    if [ "$b" != "noop" ]; then
        for g1 in ${!gpus}; do
            for g2 in ${!gpus}; do
                if [ "$g2" != "$g1" ]; then
                    eval "$BENCH" --benchmark_filter="$b/.*/$g1/$g2/" "$BENCHMARK_OUT"/`hostname`-$b-$g1-$g2.json --benchmark_repetitions=5;
                fi
            done
        done
    fi
done

for b in "${numa_gpu_gpu_bmarks[@]}"; do
    if [ "$b" != "noop" ]; then
        for n in ${!numas}; do
            for g1 in ${!gpus}; do
                for g2 in ${!gpus}; do
                    if [ "$g2" != "$g1" ]; then
                        eval "$BENCH" --benchmark_filter="$b/.*/$n/$g1/$g2/" "$BENCHMARK_OUT"/`hostname`-$b-$n-$g1-$g2.json --benchmark_repetitions=5;
                    fi
                done
            done
        done
    fi
done
