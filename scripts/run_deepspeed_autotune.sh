#!/usr/bin/env bash
# DeepSpeed AutoTuning launch script to discover optimal bucket sizes & ZeRO offload parameters

echo "[AUTOTUNE] Starting DeepSpeed AutoTuning session..."
deepspeed --autotune src/train.py \
    --deepspeed \
    --deepspeed_config configs/ds_config_zero2.json \
    --batch_size 32 \
    --lr 3e-4 \
    --epochs 1
