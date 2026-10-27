import torch

class FP8Quantizer:
    """
    FP8 (E4M3 / E5M2) quantization for sub-byte gradient accumulation.
    """
    def __init__(self, format_type: str = "e4m3fn"):
        self.dtype = torch.float8_e4m3fn if format_type == "e4m3fn" else torch.float8_e5m2

    def quantize(self, tensor: torch.Tensor):
        scale = tensor.abs().max() / torch.finfo(self.dtype).max
        scale = torch.clamp(scale, min=1e-8)
        quantized = (tensor / scale).to(self.dtype)
        return quantized, scale

    def decompress(self, quantized: torch.Tensor, scale: torch.Tensor):
        return quantized.to(torch.float32) * scale
