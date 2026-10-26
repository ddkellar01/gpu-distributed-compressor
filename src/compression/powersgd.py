import torch
import torch.distributed as dist

class PowerSGDCommunicator:
    """
    PowerSGD Gradient Compression Hook for PyTorch DDP / Custom AllReduce.
    Compresses gradient tensors using low-rank matrix decomposition.
    """
    def __init__(self, rank_matrix=4, group=None):
        self.matrix_rank = rank_matrix
        self.group = group or dist.group.WORLD

    def compress_and_allreduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim < 2:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.group)
            tensor /= dist.get_world_size(self.group)
            return tensor

        shape = tensor.shape
        matrix = tensor.view(shape[0], -1)
        m, n = matrix.shape
        r = min(self.matrix_rank, m, n)

        q = torch.randn(n, r, device=tensor.device, dtype=tensor.dtype)
        p = torch.matmul(matrix, q)
        dist.all_reduce(p, op=dist.ReduceOp.SUM, group=self.group)

        q_orth, _ = torch.linalg.qr(q)
        decompressed = torch.matmul(p, q_orth.T).view(shape)
        decompressed /= dist.get_world_size(self.group)
        
        tensor.copy_(decompressed)
        return tensor
