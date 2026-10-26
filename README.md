# GPU Distributed Data Compressor & Accelerator

A high-performance distributed training framework optimized for massive multi-node GPU clusters. It addresses network bottlenecks via **PowerSGD low-rank gradient compression** and **Top-K/INT8 quantization**, while stabilizing convergence at scale using **Linear Warmup LR schedules** and **Gradient Accumulation**.

## Quickstart

1. Install requirements:
   ```bash
   pip install -r requirements.txt
