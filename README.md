# GPU Distributed Compressor

An optimized library for high-throughput multi-GPU and multi-node LLM training utilizing low-rank tensor compression (PowerSGD), Top-K / INT8 gradient quantization, and DeepSpeed ZeRO integration.

## Folder Hierarchy

```text
gpu-distributed-compressor/
├── configs/
│   ├── ds_config_zero2.json
│   └── ds_config_zero3_offload.json
├── scripts/
│   ├── run_multi_node.sh
│   └── run_single_node.sh
├── src/
│   ├── compression/
│   │   ├── __init__.py
│   │   ├── powersgd.py
│   │   └── quantizer.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── model_builder.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── distributed.py
│   │   └── lr_scheduler.py
│   ├── __init__.py
│   └── train.py
├── README.md
└── requirements.txt
