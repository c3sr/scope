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

if [[ ${DO_BUILD} == 1 ]]; then
    echo "DO_BUILD == 1, not installing rai"
    exit 0
fi

if [[ ${DOCKER_ARCH} != ppc64le ]]; then
    echo "DOCKER_ARCH != ppc64le, not installing rai"
    exit 0
fi

mkdir -p ${RAI_ROOT}
or_die wget -q https://github.com/rai-project/rai/releases/download/v0.2.57/linux-amd64.tar.gz
or_die tar -xvf linux-amd64.tar.gz -C ${RAI_ROOT}
chmod +x ${RAI_ROOT}/rai

set +x
exit 0