import torch
import torch.distributed as dist

def compressed_allreduce(tensor: torch.Tensor, compressor, group=None):
    """
    Custom distributed AllReduce using lossy gradient compression over raw sockets.
    """
    if not dist.is_initialized():
        return tensor

    compressed_data, shape = compressor.compress("allreduce", tensor)
    
    if isinstance(compressed_data, tuple):
        p, q = compressed_data
        dist.all_reduce(p, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(q, op=dist.ReduceOp.SUM, group=group)
        p /= dist.get_world_size(group)
        q /= dist.get_world_size(group)
        return compressor.decompress((p, q), shape)
    else:
        dist.all_reduce(compressed_data, op=dist.ReduceOp.SUM, group=group)
        compressed_data /= dist.get_world_size(group)
        return compressed_data
