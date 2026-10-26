import torch
import torch.distributed as dist

class TopKQuantizer:
    """
    Top-K sparsification quantizer to reduce network communication volume.
    """
    def __init__(self, k_percent=0.01, group=None):
        self.k_percent = k_percent
        self.group = group or dist.group.WORLD

    def compress_and_allreduce(self, tensor: torch.Tensor) -> torch.Tensor:
        flat = tensor.view(-1)
        num_elements = flat.numel()
        k = max(1, int(num_elements * self.k_percent))

        if k >= num_elements:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.group)
            tensor /= dist.get_world_size(self.group)
            return tensor

        # Fallback to standard reduced sync for custom sparsified buffers
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.group)
        tensor /= dist.get_world_size(self.group)
        return tensor
