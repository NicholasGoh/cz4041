#!/bin/bash

docker run \
    --rm \
    --gpus all \
    -it \
    --shm-size=8gb \
    -v=$PWD/ai/tron2/weights:/tmp \
    -v=$PWD/ai/actions:/home/nic/detectron2/actions \
    -v=$PWD/ai/loaders:/home/nic/detectron2/loaders \
    -v=$PWD/ai/utils:/home/nic/detectron2/utils \
    -v=/data:/data \
    tron2

echo -ne '\007'
    # -v=$PWD/ai/actions:$HOME/detectron2/actions \
