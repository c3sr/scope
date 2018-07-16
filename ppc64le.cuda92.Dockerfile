FROM nvidia/cuda-ppc64le:9.2-cudnn7-devel

RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
    curl \
    git \
    libnuma-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://cmake.org/files/v3.11/cmake-3.11.4.tar.gz -o cmake.tar.gz \
    $$ mkdir -p cmake \
    && tar --strip-components=1 -C cmake -xf cmake.tar.gz \
    && cd cmake \
    && ./bootstrap  && make && make install \
    && cd .. \
    && rm -r cmake \
    && rm cmake.tar.gz

RUN cmake --version
RUN gcc --version

COPY . microbench

WORKDIR microbench

RUN mkdir -p build \
    && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && make VERBOSE=1

RUN mv build/bench /bin/.

