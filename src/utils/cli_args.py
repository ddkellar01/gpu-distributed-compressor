import argparse

def parse_cli_args():
    parser = argparse.ArgumentParser(description="Distributed Deep Learning GPU Compression Engine")
    parser.add_argument("--config", type=str, default="configs/ds_config_zero2.json", help="DeepSpeed JSON configuration path")
    parser.add_argument("--world_size", type=int, default=8, help="Number of active GPU ranks")
    parser.add_argument("--batch_size", type=int, default=32, help="Micro batch size per process")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--rank_compression", type=int, default=4, help="PowerSGD low-rank factor")
    parser.add_argument("--enable_triton", action="store_true", help="Enable custom Triton GPU kernels")
    return parser.parse_args()
