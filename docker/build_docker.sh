#!/bin/bash

docker build --build-arg UID=$(id -u) \
    --build-arg GID=$(id -g) \
    -t cz4041 \
    -f $PWD/docker/Dockerfile \
    .
