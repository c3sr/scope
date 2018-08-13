FROM nvidia/cuda-ppc64le:9.2-cudnn7-devel

RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
    curl \
    git \
    libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://cmake.org/files/v3.12/cmake-3.12.1.tar.gz -o cmake.tar.gz \
    $$ mkdir -p cmake \
    && tar --strip-components=1 -C cmake -xf cmake.tar.gz \
    && cd cmake \
    && ./bootstrap  && make && make install \
    && cd .. \
    && rm -r cmake \
    && rm cmake.tar.gz

RUN cmake --version
RUN gcc --version

COPY . scope
WORKDIR scope

RUN mkdir -p build \
    && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_MISC=OFF \
    -DENABLE_NCCL=OFF \
    -DNVCC_ARCH_FLAGS="3.0 3.2 3.5 3.7 5.0 5.2 5.3 6.0" \
    && make VERBOSE=1

RUN mv build/scope /bin/.
