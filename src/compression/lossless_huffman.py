import torch

class SparseThresholdCompressor:
    """
    Combines hard gradient thresholding with zero-run elimination for ultra-high compression ratios.
    """
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold

    def compress(self, tensor: torch.Tensor):
        mask = torch.abs(tensor) >= self.threshold
        indices = torch.nonzero(mask.view(-1)).squeeze(-1)
        values = tensor.view(-1)[indices]
        return (values, indices), tensor.shape

    def decompress(self, compressed_tuple):
        (values, indices), original_shape = compressed_tuple
        flat_size = original_shape.numel()
        decompressed = torch.zeros(flat_size, device=values.device, dtype=values.dtype)
        decompressed[indices] = values
        return decompressed.view(original_shape)
