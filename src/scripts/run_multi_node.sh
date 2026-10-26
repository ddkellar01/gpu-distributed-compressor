#!/bin/bash

# Multi-Node Distributed Training Launcher
NNODES=4
NODE_RANK=0
MASTER_ADDR="192.168.1.100"
MASTER_PORT=29500
GPUS_PER_NODE=8

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/train.py \
    --batch_size 8 \
    --grad_accum_steps 16 \
    --lr 0.001 \
    --epochs 10
