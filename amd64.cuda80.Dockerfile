FROM nvidia/cuda:8.0-cudnn7-devel

RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
    curl \
    git \
    libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://cmake.org/files/v3.12/cmake-3.12.1-Linux-x86_64.tar.gz -o cmake.tar.gz \
    && tar -xf cmake.tar.gz \
    && cp -r cmake-3.12.1-Linux-x86_64/* /usr/. \
    && rm cmake.tar.gz

RUN cmake --version
RUN gcc --version

COPY . microbench

WORKDIR microbench

RUN mkdir -p build \
    && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_MISC=OFF \
    -DENABLE_NCCL=OFF \
    -DNVCC_ARCH_FLAGS="2.0 3.0 3.2 3.5 3.7 5.0 5.2 5.3" \
    && make VERBOSE=1

RUN mv build/bench /bin/.

