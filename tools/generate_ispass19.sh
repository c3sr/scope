#! /bin/bash

# h2d and d2h
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_Duplex_NUMAMemcpy_HostToGPU/0   > run_Comm_Duplex_NUMAMemcpy_HostToGPU_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_Duplex_NUMAMemcpy_GPUToHost/0   > run_Comm_Duplex_NUMAMemcpy_GPUToHost_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_Duplex_NUMAMemcpy_PinnedToGPU/0 > run_Comm_Duplex_NUMAMemcpy_PinnedToGPU_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_Duplex_NUMAMemcpy_GPUToPinned/0 > run_Comm_Duplex_NUMAMemcpy_GPUToPinned_0.sh

# h2d and d2h flush
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_Duplex_NUMAMemcpy_HostToGPU_flush/0   > run_Comm_Duplex_NUMAMemcpy_HostToGPU_flush_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_Duplex_NUMAMemcpy_GPUToHost_flush/0   > run_Comm_Duplex_NUMAMemcpy_GPUToHost_flush_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_Duplex_NUMAMemcpy_PinnedToGPU_flush/0 > run_Comm_Duplex_NUMAMemcpy_PinnedToGPU_flush_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_Duplex_NUMAMemcpy_GPUToPinned_flush/0 > run_Comm_Duplex_NUMAMemcpy_GPUToPinned_flush_0.sh

# coherence
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_UM_Coherence_HostToGPU/0 > run_Comm_UM_Coherence_HostToGPU_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_UM_Coherence_GPUToHost/0 > run_Comm_UM_Coherence_GPUToHost_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_UM_Coherence_GPUToGPU    > run_Comm_UM_Coherence_GPUToGPU.sh

# unified
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_UM_Prefetch_HostToGPU/0 > run_Comm_UM_Prefetch_HostToGPU_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_UM_Prefetch_GPUToHost/0 > run_Comm_UM_Prefetch_GPUToHost_0.sh
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_UM_Prefetch_GPUToGPU    > run_Comm_UM_Prefetch_GPUToGPU.sh

# d2d duplex peer
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_Duplex_Memcpy_GPUGPUPeer > run_Comm_Duplex_Memcpy_GPUGPUPeer.sh

# d2d peer
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_Memcpy_GPUToGPUPeer > run_Comm_Memcpy_GPUToGPUPeer.sh

# h2d pageable duplex
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_Duplex_NUMAMemcpy_Host/0 > run_Comm_Duplex_NUMAMemcpy_Host_0.sh

# h2d pinned duplex
./generate_runscripts.py --benchmark-repetitions=5 --benchmark-filter=Comm_Duplex_NUMAMemcpy_Pinned/0 > run_Comm_Duplex_NUMAMemcpy_Pinned_0.sh
