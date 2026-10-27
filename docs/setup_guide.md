# Deployment and Setup Guide

## System Requirements
- NVIDIA Drivers >= 525.60.13
- CUDA >= 12.1
- PyTorch >= 2.1.0
- DeepSpeed >= 0.12.0

## Quickstart

```bash
# Clone and install dependencies
git clone [https://github.com/ddkellar01/gpu-distributed-compressor.git](https://github.com/ddkellar01/gpu-distributed-compressor.git)
cd gpu-distributed-compressor
pip install -r requirements.txt

# Run test suite
pytest tests/

# Execute single-node 8 GPU run
bash scripts/run_single_node.sh 8 configs/ds_config_zero2.json
