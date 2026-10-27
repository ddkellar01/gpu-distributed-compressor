import os

def configure_nccl_environment(enable_nvlink: bool = True, num_channels: int = 16):
    """
    Sets optimized environment flags for NCCL cross-node socket & P2P throughput tuning.
    """
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_MIN_NCHANNELS"] = str(num_channels)
    os.environ["NCCL_IB_DISABLE"] = "0" if enable_nvlink else "1"
    os.environ["NCCL_NET_GDR_LEVEL"] = "5"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    print(f"[NCCL TUNER] Enabled NCCL tuning with {num_channels} channels and GDR Level 5.")
