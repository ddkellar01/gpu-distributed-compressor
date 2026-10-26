# GPU Distributed Data Compressor & Accelerator

A high-performance distributed training framework optimized for massive multi-node GPU clusters. It addresses network bottlenecks via **PowerSGD low-rank gradient compression** and **Top-K/INT8 quantization**, while stabilizing convergence at scale using **Linear Warmup LR schedules** and **Gradient Accumulation**.

## Quickstart

1. Install requirements:
   ```bash
   pip install -r requirements.txt

gpu-distributed-compressor/
├── configs/
│   ├── ds_config_zero2.json             # DeepSpeed ZeRO-2 cluster configuration
│   └── ds_config_zero3_offload.json     # DeepSpeed ZeRO-3 with CPU/NVMe offloading
├── scripts/
│   ├── run_multi_node.sh                # Multi-node cluster launch script (SLURM / torchrun)
│   └── run_single_node.sh               # Single-node multi-GPU launch script (8x GPUs)
├── src/
│   ├── __init__.py
│   ├── compression/
│   │   ├── __init__.py
│   │   ├── powersgd.py                  # PowerSGD low-rank gradient compression hook
│   │   └── quantizer.py                 # Top-K & INT8 quantization mechanisms
│   ├── models/
│   │   ├── __init__.py
│   │   └── model_builder.py             # Backbone transformer / model constructor
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── distributed.py               # NCCL environment initialization & rank management
│   │   └── lr_scheduler.py              # Linear warmup & cosine annealing learning rate scheduler
│   └── train.py                         # Main distributed execution entry point
├── requirements.txt                     # Python dependencies (PyTorch, DeepSpeed, Triton, etc.)
└── README.md                            # Comprehensive documentation and setup guide
