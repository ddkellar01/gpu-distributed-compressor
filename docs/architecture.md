# Project Architecture: GPU Distributed Compressor

## Core Engine Components
- **`src/compression/`**: Implements low-rank matrix decomposition (PowerSGD), quantized formats (INT8, FP8), and adaptive Triton GPU kernels.
- **`src/models/`**: Parameterizable Transformer backbone and LoRA injection layers.
- **`src/utils/`**: Distributed process initialization, learning rate annealing, Prometheus metrics, and PyTorch Profiler traces.
- **`configs/`**: DeepSpeed ZeRO-1, ZeRO-2, and ZeRO-3 JSON configurations.
- **`scripts/`**: Distributed benchmark execution, bandwidth testing, ONNX export, and hyperparameter tuning.
