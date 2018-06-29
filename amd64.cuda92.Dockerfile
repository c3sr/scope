FROM nvidia/cuda:9.2-cudnn7-devel

RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
    curl \
    git \
    libnuma-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://cmake.org/files/v3.11/cmake-3.11.2-Linux-x86_64.tar.gz -o cmake.tar.gz \
    && tar -xf cmake.tar.gz \
    && cp -r cmake-3.11.2-Linux-x86_64/* /usr/. \
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

