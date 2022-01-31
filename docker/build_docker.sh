#!/bin/bash

docker build --build-arg UID=$(id -u) \
    --build-arg GID=$(id -g) \
    -t tron2 \
    -f $PWD/docker/Dockerfile \
    .
