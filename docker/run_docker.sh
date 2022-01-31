#!/bin/bash

docker run \
    --rm \
    --gpus all \
    -it \
    --shm-size=8gb \
    -v=$PWD/ai/tron2/weights:/tmp \
    -v=$PWD/ai/actions:$HOME/detectron2/actions \
    -v=$PWD/ai/loaders:$HOME/detectron2/loaders \
    -v=$PWD/ai/utils:$HOME/detectron2/utils \
    -v=$PWD/data:/data \
    tron2

echo -ne '\007'
