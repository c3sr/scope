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

if [ $CI ]; then
    source ~/.bashrc
    cd ${TRAVIS_BUILD_DIR}
    BRANCH=$TRAVIS_BRANCH
else
    BRANCH=`git rev-parse --abbrev-ref HEAD`
fi

# replace / with -
BRANCH="${BRANCH//\//-}"


# if DOCKER_ARCH is set
if [[ ! -z ${DOCKER_ARCH+x} ]]; then
    ARCH=${DOCKER_ARCH}
else
    ARCH=`uname -m`
    if [ $ARCH == x86_64 ]; then
        ARCH=amd64
    fi
fi

REPO=raiproject/scope
TAG=`if [ "$BRANCH" == "master" ]; then echo "latest"; else echo "${BRANCH}"; fi`

echo "$REPO"
echo "$TAG"

# untracked files
git ls-files --exclude-standard --others
DIRTY=$?

if [ "$DIRTY" == 0 ]; then
# staged changes, not yet committed
git diff-index --quiet --cached HEAD --
DIRTY=$?
fi

if [ "$DIRTY" == 0 ]; then
# working tree has changes that could be staged
git diff-files --quiet
DIRTY=$?
fi

if [ "$DIRTY" != 0 ]; then
    TAG=$TAG-dirty
fi


if [[ ! -z ${DOCKER_ARCH+x} ]]; then
    set +x
    echo "$DOCKER_PASSWORD" | or_die docker login --username "$DOCKER_USERNAME" --password-stdin
    set -x

    if [ "$ARCH" == amd64 ]; then # if amd64, build on travis
        or_die docker build -f $ARCH.cuda${DOCKER_CUDA}.Dockerfile -t $REPO:$ARCH-cuda${DOCKER_CUDA}-$TAG .
        or_die docker push $REPO
    elif [ "$ARCH" == ppc64le ]; then # if ppc64le, build on rai
        cp -v ci/rai/.rai_profile $HOME/.rai_profile
        echo "  secret_key: ${RAI_SECRET_KEY}" >> $HOME/.rai_profile
        mv -v ci/rai/rai_build.cuda${DOCKER_CUDA}.yml rai_build.yml
        rai -p .
        or_die docker push $REPO
    fi
fi

set +x
exit 0