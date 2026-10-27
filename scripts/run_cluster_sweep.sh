#!/usr/bin/env bash
# Parameter sweep script for hyperparameter optimization across Slurm nodes

LEARNING_RATES=(1e-5 3e-5 1e-4 3e-4)
RANKS=(2 4 8 16)

for lr in "${LEARNING_RATES[@]}"; do
  for rank in "${RANKS[@]}"; do
    echo "[SWEEP] Launching trial with lr=$lr, rank=$rank"
    python src/train.py --lr "$lr" --rank_compression "$rank"
  done
done
