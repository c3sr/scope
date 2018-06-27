FROM nvidia/cuda:9.2-cudnn7-devel

RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://cmake.org/files/v3.11/cmake-3.11.2-Linux-x86_64.tar.gz -o cmake.tar.gz \
    && tar -xf cmake.tar.gz \
    && cp -r cmake-3.11.2-Linux-x86_64/* /usr/. \
    && rm cmake.tar.gz

RUN cmake --version

ENV SUGAR_ROOT ${HOME}/.sugar
ENV CUB_ROOT ${HOME}/.cub
ENV SPDLOG_ROOT ${HOME}/.spdlog
ENV BECNHMARK_ROOT ${HOME}/.benchmark
ENV FMT_ROOT ${HOME}/.fmt

# Install sugar
RUN curl -sSL https://github.com/ruslo/sugar/archive/v1.3.0.tar.gz -o sugar.tar.gz \
    && tar -xf sugar.tar.gz \
    && cp -r v1.3.0/* ${SUGAR_ROOT}/. \
    && rm sugar.tar.gz

# install cub
RUN curl -sSL https://github.com/NVlabs/cub/archive/v1.8.0.tar.gz -o cub.tar.gz \
    && tar -xf cub.tar.gz \
    && cp -r v1.8.0/* ${CUB_ROOT}/. \
    && rm cub.tar.gz

# install spdlog
RUN curl -sSL https://github.com/gabime/spdlog/archive/v0.17.0.tar.gz -o spdlog.tar.gz \
    && tar -xf spdlog.tar.gz \
    && cp -r v0.17.0/* ${SPDLOG_ROOT}/. \
    && rm spdlog.tar.gz

# install benchmark
RUN curl -sSL https://github.com/google/benchmark/archive/v1.4.1.tar.gz -o benchmark.tar.gz \
    && tar -xf benchmark.tar.gz \
    && cp -r v1.4.1/* ${BENCHMARK_ROOT}/. \
    && rm benchmark.tar.gz

# install fmt
RUN curl -sSL https://github.com/fmtlib/fmt/archive/5.0.0.tar.gz -o fmt.tar.gz \
    && tar -xf fmt.tar.gz \
    && cp -r v5.0.0/* ${FMT_ROOT}/. \
    && rm fmt.tar.gz

COPY . microbench

WORKDIR microbench

RUN mkdir -p build \
    && cd build \
    && cmake .. \
    -DCONFIG_USE_HUNTER=OFF \
    -DSUGAR_ROOT=${SUGAR_ROOT} \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA_EVENTS=ON \
    -DNVCC_ARCH_FLAGS="2.0 3.0 3.2 3.5 3.7 5.0 5.2 5.3" \
    && make VERBOSE=1

RUN mv build/bench /bin/.

