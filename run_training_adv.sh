#!/bin/bash

# Set the GPU to use (GPU with original ID 2)
#export CUDA_VISIBLE_DEVICES="3"

# Run the training script
python main.py "$@"
