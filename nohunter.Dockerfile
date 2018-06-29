FROM nvidia/cuda:9.2-cudnn7-devel

RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
    curl \
    libnuma-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://cmake.org/files/v3.11/cmake-3.11.4-Linux-x86_64.tar.gz -o cmake.tar.gz \
    && mkdir -p cmake \
    && tar -xf cmake.tar.gz --strip-components=1 -C cmake \
    && cp -r cmake/* /usr/. \
    && rm cmake.tar.gz

RUN cmake --version

ENV HOME /opt
ENV BENCHMARK_ROOT ${HOME}/.benchmark
ENV CUB_ROOT ${HOME}/.cub
ENV FMT_ROOT ${HOME}/.fmt
ENV GTEST_ROOT ${HOME}/.gtest
ENV SPDLOG_ROOT ${HOME}/.spdlog
ENV SUGAR_ROOT ${HOME}/.sugar

# build gtest
RUN cd $HOME \
    && curl -sSL https://github.com/google/googletest/archive/release-1.8.0.tar.gz -o gtest.tar.gz \
    && mkdir -p gtest/build \
    && tar -xf gtest.tar.gz --strip-components=1 -C gtest \
    && cd gtest/build \
    && cmake $HOME/gtest -DCMAKE_INSTALL_PREFIX=${GTEST_ROOT} \
    && make -j4 install \
    && cd ${HOME}

# install benchmark
RUN cd $HOME \
    && curl -sSL https://github.com/google/benchmark/archive/v1.4.1.tar.gz -o benchmark.tar.gz \
    && mkdir -p benchmark/build \
    && tar -xf benchmark.tar.gz --strip-components=1 -C benchmark \
    && cd benchmark/build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=${BENCHMARK_ROOT} -DGTEST_ROOT=${GTEST_ROOT} \
    && make -j4 install \
    && cd $HOME \
    && rm benchmark.tar.gz

# Install sugar
RUN cd $HOME \
    && curl -sSL https://github.com/ruslo/sugar/archive/v1.3.0.tar.gz -o sugar.tar.gz \
    && mkdir -p sugar \
    && tar -xf sugar.tar.gz --strip-components=1 -C sugar \
    && cd sugar \
    && mkdir -p build \
    && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=${SUGAR_ROOT} .. \
    && make -j4 install \
    && cd $HOME \
    && rm sugar.tar.gz

# install cub
RUN cd $HOME \
    && curl -sSL https://github.com/NVlabs/cub/archive/v1.8.0.tar.gz -o cub.tar.gz \
    && mkdir -p cub \
    && tar -xf cub.tar.gz --strip-components=1 -C cub \
    && mkdir -p ${CUB_ROOT} \
    && cp -rv cub/* ${CUB_ROOT}/. \
    && rm cub.tar.gz

# install spdlog
RUN cd $HOME \
    && curl -sSL https://github.com/gabime/spdlog/archive/v0.17.0.tar.gz -o spdlog.tar.gz \
    && mkdir -p spd/build \
    && tar -xf spdlog.tar.gz --strip-components=1 -C spd \
    && cd spd/build \ 
    && cmake .. -DCMAKE_INSTALL_PREFIX=${SPDLOG_ROOT} \
    && make -j4 install \
    && cd $HOME \
    && rm spdlog.tar.gz

# install fmt 4.0
RUN cd $HOME \
    && curl -sSL https://github.com/fmtlib/fmt/archive/4.0.0.tar.gz -o fmt.tar.gz \
    && mkdir -p fmt/build \
    && tar -xf fmt.tar.gz --strip-components=1 -C fmt \
    && cd fmt/build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=${FMT_ROOT} \
    && make -j4 install \
    && cd $HOME \
    && rm fmt.tar.gz

COPY . microbench

WORKDIR microbench

RUN mkdir -p build \
    && cd build \
    && cmake .. \
    -DCONFIG_USE_HUNTER=OFF \
    -DCMAKE_PREFIX_PATH="${BENCHMARK_ROOT};${CUB_ROOT};${FMT_ROOT};${SPDLOG_ROOT};${BENCHMARK_ROOT}" \
    -DSUGAR_ROOT=${SUGAR_ROOT} \
    -DCMAKE_BUILD_TYPE=Release \
    -DNVCC_ARCH_FLAGS="3.0 3.2 3.5 3.7 5.0 5.2 5.3" \
    && make -j4 || make VERBOSE=1

RUN mv build/bench /bin/.

