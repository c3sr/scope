#! /bin/bash

set -x

function or_die () {
    "$@"
    local status=$?
    if [[ $status != 0 ]] ; then
        echo ERROR $status command: $@
        exit $status
    fi
}

source ~/.bashrc

if [[ ! -z ${CMAKE_VERSION+x} ]]; then
    echo "Installing CMake"
    v=( ${CMAKE_VERSION//./ } )  # replace points, split into array
    MAJOR_MINOR="${a[0]}.${a[1]}" # only major and minor version
    mkdir -p ${CMAKE_ROOT}
    cd ${CMAKE_ROOT}
    wget -q https://cmake.org/files/v${MAJOR_MINOR}/cmake-${CMAKE_VERSION}-Linux-x86_64.sh
    sh cmake-${CMAKE_VERSION}-Linux-x86_64.sh --prefix=${CMAKE_ROOT} --exclude-subdir
    ls -l ${CMAKE_ROOT}
fi

set +x
exit 0