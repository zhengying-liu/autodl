#!/bin/bash
set -ex

USERNAME=evariste
IMAGE=autodl
VERSION=`cat VERSION`

docker push $USERNAME/$IMAGE:gpu-$VERSION
docker push $USERNAME/$IMAGE:gpu
docker push $USERNAME/$IMAGE:cpu
