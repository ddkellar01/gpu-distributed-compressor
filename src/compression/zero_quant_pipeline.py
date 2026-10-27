import torch

class ZeroQuantPipeline:
    """
    Hardware-aligned 8-bit dynamic quantization pipeline for layer weights and activation gradients.
    """
    def __init__(self, token_granularity: bool = True):
        self.token_granularity = token_granularity

    def quantize_activations(self, act: torch.Tensor):
        if self.token_granularity and act.ndim == 3:
            # Per-token scale calculation along feature dimension
            scales = torch.max(torch.abs(act), dim=-1, keepdim=True).values / 127.0
            scales = torch.clamp(scales, min=1e-8)
            quantized = torch.clamp(torch.round(act / scales), -128, 127).to(torch.int8)
            return quantized, scales
        else:
            scale = torch.max(torch.abs(act)) / 127.0
            quantized = torch.clamp(torch.round(act / scale), -128, 127).to(torch.int8)
            return quantized, scale
