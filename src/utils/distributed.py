import os
import torch
import torch.distributed as dist

def init_distributed_env(backend="nccl"):
    """
    Initializes the distributed environment via NCCL.
    """
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()
