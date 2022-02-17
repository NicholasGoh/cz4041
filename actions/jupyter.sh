#!/bin/bash

export TF_FORCE_GPU_ALLOW_GROWTH=true
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
