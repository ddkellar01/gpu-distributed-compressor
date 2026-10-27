#!/bin/bash

# Single-Node Multi-GPU Training Launcher (8x GPUs)
GPUS_PER_NODE=8

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    src/train.py \
    --batch_size 16 \
    --grad_accum_steps 4 \
    --lr 1e-3 \
    --epochs 5
