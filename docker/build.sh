#!/bin/bash
set -ex

USERNAME=evariste
IMAGE=autodl
VERSION=`cat VERSION`

# Download embedding weights files
curl -C - -O https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.vec.gz
curl -C - -O https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
curl -C - -O https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip

docker build -t $USERNAME/$IMAGE:gpu-$VERSION .
docker tag $USERNAME/$IMAGE:gpu-$VERSION $USERNAME/$IMAGE:gpu-latest
docker build -t $USERNAME/$IMAGE:cpu-$VERSION -f Dockerfile.cpu .
docker tag $USERNAME/$IMAGE:cpu-$VERSION $USERNAME/$IMAGE:cpu-latest
