#! /bin/bash

set -xeuo pipefail

# h2d and d2h
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_NUMAMemcpy_HostToGPU/0   > run_Comm_NUMAMemcpy_HostToGPU_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_NUMAMemcpy_GPUToHost/0   > run_Comm_NUMAMemcpy_GPUToHost_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_NUMAMemcpy_PinnedToGPU/0 > run_Comm_NUMAMemcpy_PinnedToGPU_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_NUMAMemcpy_GPUToPinned/0 > run_Comm_NUMAMemcpy_GPUToPinned_0.sh

# h2d and d2h flush
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_NUMAMemcpy_HostToGPU_flush/0   > run_Comm_NUMAMemcpy_HostToGPU_flush_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_NUMAMemcpy_GPUToHost_flush/0   > run_Comm_NUMAMemcpy_GPUToHost_flush_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_NUMAMemcpy_PinnedToGPU_flush/0 > run_Comm_NUMAMemcpy_PinnedToGPU_flush_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_NUMAMemcpy_GPUToPinned_flush/0 > run_Comm_NUMAMemcpy_GPUToPinned_flush_0.sh

#d2d nopeer
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_NUMAMemcpy_GPUToGPU/0 > run_Comm_NUMAMemcpy_GPUToGPU_0.sh

#h2h
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_NUMAMemcpy_HostToPinned/0 > run_Comm_NUMAMemcpy_HostToPinned_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_NUMAMemcpy_HostToPinned_flush/0 > run_Comm_NUMAMemcpy_HostToPinned_flush_0.sh

# h2d duplex
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_Duplex_NUMAMemcpy_Host/0   > run_Comm_Duplex_NUMAMemcpy_Host_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_Duplex_NUMAMemcpy_Pinned/0 > run_Comm_Duplex_NUMAMemcpy_Pinned_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_Duplex_NUMAMemcpy_Host_flush/0   > run_Comm_Duplex_NUMAMemcpy_Host_flush_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_Duplex_NUMAMemcpy_Pinned_flush/0 > run_Comm_Duplex_NUMAMemcpy_Pinned_flush_0.sh

# coherence
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_UM_Coherence_HostToGPU/0 > run_Comm_UM_Coherence_HostToGPU_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_UM_Coherence_GPUToHost/0 > run_Comm_UM_Coherence_GPUToHost_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_UM_Coherence_GPUToHostMt/0 > run_Comm_UM_Coherence_GPUToHostMt_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_UM_Coherence_GPUToGPU/0    > run_Comm_UM_Coherence_GPUToGPU_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_Coherence_Duplex_GPUGPU/0  > run_Comm_Coherence_Duplex_GPUGPU_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_Coherence_Duplex_HostGPU/0 > run_Comm_Coherence_Duplex_HostGPU_0.sh


# unified
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_UM_Prefetch_HostToGPU/0 > run_Comm_UM_Prefetch_HostToGPU_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_UM_Prefetch_GPUToHost/0 > run_Comm_UM_Prefetch_GPUToHost_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_UM_Prefetch_GPUToGPU/0    > run_Comm_UM_Prefetch_GPUToGPU_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_Prefetch_Duplex_GPUGPU/0  > run_Comm_Prefetch_Duplex_GPUGPU_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_Prefetch_Duplex_HostGPU/0 > run_Comm_Prefetch_Duplex_HostGPU_0.sh


# d2d duplex peer
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_Duplex_Memcpy_GPUGPUPeer > run_Comm_Duplex_Memcpy_GPUGPUPeer.sh

# d2d peer
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_Memcpy_GPUToGPUPeer > run_Comm_Memcpy_GPUToGPUPeer.sh

# zero-copy
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_ZeroCopy_HostToGPU_Read/0  > run_Comm_ZeroCopy_HostToGPU_Read_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_ZeroCopy_HostToGPU_Write/0 > run_Comm_ZeroCopy_HostToGPU_Write_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_ZeroCopy_GPUToGPU_Read/0  > run_Comm_ZeroCopy_GPUToGPU_Read_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_ZeroCopy_GPUToGPU_Write/0 > run_Comm_ZeroCopy_GPUToGPU_Write_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_ZeroCopy_Duplex_GPUGPU_Read/0  > run_Comm_ZeroCopy_Duplex_GPUGPU_Read_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_ZeroCopy_Duplex_GPUGPU_Write/0 > run_Comm_ZeroCopy_Duplex_GPUGPU_Write_0.sh

