#! env bash

# Run all NUMA benchmarks in their own process

set -eou pipefail -x

machine=$1

s822lc_gpus=(
0
1
2
3
)

s822lc_numas=(
0
1
)

ac922_gpus=(
0
1
2
3
)

ac922_numas=(
0
8
)

dgx_gpus=(
0
1
2
3
4
5
6
7
)

dgx_numas=(
0
1
)

numas=$machine\_numas[@]
gpus=$machine\_gpus[@]

numa_gpu_gpu_bmarks=(
NUMA_Memcpy_GPUToGPU
)

gpu_gpu_bmarks=(
UM_Coherence_GPUToGPU
UM_Prefetch_GPUToGPU
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
)

#for b in "${numa_gpu_bmarks[@]}"; do
#    for n in ${!numas}; do
#        for g in ${!gpus}; do
#            ./bench --benchmark_filter="$b.*/$n/$g/" --benchmark_out=`hostname`-$b-$n-$g.json --benchmark_repetitions=5;
#        done
#    done
#done
#
#for b in "${gpu_gpu_bmarks[@]}"; do
#    for g1 in ${!gpus}; do
#        for g2 in ${!gpus}; do
#                if [ "$g2" != "$g1" ]; then
#                    ./bench --benchmark_filter="$b.*/$g1/$g2/" --benchmark_out=`hostname`-$b-$g1-$g2.json --benchmark_repetitions=5;
#                fi
#        done
#    done
#done



for b in "${numa_gpu_gpu_bmarks[@]}"; do
    for n in ${!numas}; do
        for g1 in ${!gpus}; do
            for g2 in ${!gpus}; do
                if [ "$g2" != "$g1" ]; then
                    ./bench --benchmark_filter="$b.*/$n/$g1/$g2/" --benchmark_out=`hostname`-$b-$n-$g1-$g2.json --benchmark_repetitions=5;
                fi
            done
        done
    done
done
