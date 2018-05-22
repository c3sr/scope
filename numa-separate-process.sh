#! /bin/bash

# Run all NUMA benchmarks in their own process

set -eou pipefail -x

# Which machine to use (s822lc, ac922, or dgx)
machine=$1
# Path to bench executable
BENCH=$2

if [ -z ${3+x} ]; then
    OUT_DIR="."
else
    OUT_DIR="$3"
fi


s822lc_gpus=( 0 1 2 3 )
s822lc_numas=( 0 1 )
 
ac922_gpus=( 0 1 2 3 )
ac922_numas=( 0 8 )

dgx_gpus=( 0 1 2 3 4 5 6 7 )
dgx_numas=( 0 1 )

numas=$machine\_numas[@]
gpus=$machine\_gpus[@]


numa_gpu_gpu_bmarks=(
NUMA_Memcpy_GPUToGPU
noop
)

gpu_gpu_bmarks=(
CUDA_Memcpy_GPUToGPU
UM_Coherence_GPUToGPU
UM_Prefetch_GPUToGPU
noop
)

numa_gpu_bmarks=(
NUMA_Memcpy_GPUToHost
NUMA_Memcpy_GPUToPinned
NUMA_Memcpy_HostToGPU
NUMA_Memcpy_PinnedToGPU
NUMAUM_Coherence_GPUToHost
NUMAUM_Coherence_HostToGPU
NUMAUM_Prefetch_GPUToHost
NUMAUM_Prefetch_HostToGPU
noop
)

mkdir -p "$OUT_DIR"

for b in "${numa_gpu_bmarks[@]}"; do
    if [ "$b" != "noop" ]; then
        for n in ${!numas}; do
            for g in ${!gpus}; do
                "$BENCH" --benchmark_filter="$b.*/$n/$g/" --benchmark_out="$OUT_DIR/`hostname`-$b-$n-$g.json" --benchmark_repetitions=5;
            done
        done
    fi
done

for b in "${gpu_gpu_bmarks[@]}"; do
    if [ "$b" != "noop" ]; then
        for g1 in ${!gpus}; do
            for g2 in ${!gpus}; do
                    if [ "$g2" != "$g1" ]; then
                        "$BENCH" --benchmark_filter="$b.*/$g1/$g2/" --benchmark_out="$OUT_DIR/`hostname`-$b-$g1-$g2.json" --benchmark_repetitions=5;
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
                        "$BENCH" --benchmark_filter="$b.*/$n/$g1/$g2/" --benchmark_out="$OUT_DIR/`hostname`-$b-$n-$g1-$g2.json" --benchmark_repetitions=5;
                    fi
                done
            done
        done
    fi
done
