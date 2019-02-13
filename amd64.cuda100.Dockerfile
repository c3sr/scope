FROM nvidia/cuda:10.0-cudnn7-devel

RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN gcc --version

RUN curl -sSL https://github.com/Kitware/CMake/releases/download/v3.13.4/cmake-3.13.4-Linux-x86_64.tar.gz -o cmake.tar.gz \
    && tar -xf cmake.tar.gz \
    && cp -r cmake-3.13.4-Linux-x86_64/* /usr/. \
    && rm cmake.tar.gz

RUN cmake --version

ENV SCOPE_ROOT /opt/scope
COPY . ${SCOPE_ROOT}
WORKDIR ${SCOPE_ROOT}

RUN mkdir -p build \
    && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && make VERBOSE=1

ENV PATH ${SCOPE_ROOT}/build:$PATH
