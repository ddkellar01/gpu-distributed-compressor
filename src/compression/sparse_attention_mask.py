import torch

class SparseAttentionCompressor:
    """
    Sparsifies attention activation maps and gradients using Block-Sparse masking.
    """
    def __init__(self, block_size: int = 64, density: float = 0.25):
        self.block_size = block_size
        self.density = density

    def compress_attention_grad(self, grad: torch.Tensor):
        if grad.ndim < 3:
            return grad, None
        
        b, h, s, d = grad.shape
        num_blocks = s // self.block_size
        k = max(1, int(num_blocks * self.density))

        blocked = grad.view(b, h, num_blocks, self.block_size, d)
        block_norms = torch.norm(blocked, dim=(-2, -1))
        
        _, top_indices = torch.topk(block_norms, k, dim=-1)
        return (blocked, top_indices), grad.shape

    def decompress_attention_grad(self, compressed_tuple, shape):
        if not isinstance(compressed_tuple, tuple):
            return compressed_tuple
        (blocked, top_indices), original_shape = compressed_tuple
        decompressed = torch.zeros(original_shape, device=blocked.device, dtype=blocked.dtype)
        return decompressed
