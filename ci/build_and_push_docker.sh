#! /bin/bash

# This script should build and push docker images in travis or on development systems
# It will tag the docker image as dirty if there are
# uncommited staged changes,
# changes that could be staged
# untracked, unignored files

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



ARCH=`uname -m`

if [ $ARCH == x86_64 ]; then
    ARCH=amd64
fi

REPO=raiproject/microbench
TAG=`if [ "$BRANCH" == "master" ]; then echo "latest"; else echo "${BRANCH//\//-}"; fi`

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

# or_die docker build -f $ARCH.cuda75.Dockerfile -t $REPO:$ARCH-cuda75-$TAG .
or_die docker build -f $ARCH.cuda80.Dockerfile -t $REPO:$ARCH-cuda80-$TAG .
or_die docker build -f $ARCH.cuda92.Dockerfile -t $REPO:$ARCH-cuda92-$TAG .

set +x
echo "$DOCKER_PASSWORD" | or_die docker login --username "$DOCKER_USERNAME" --password-stdin
set -x

or_die docker push $REPO

set +x
exit 0