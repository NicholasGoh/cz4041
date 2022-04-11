#!/bin/bash

docker run \
    --rm \
    --gpus all \
    -it \
    --shm-size=8gb \
    -p 8888:8888 \
    -v=$PWD:$HOME/cz4041 \
    -v=/data/plant-seedlings-classification:/data \
    cz4041
