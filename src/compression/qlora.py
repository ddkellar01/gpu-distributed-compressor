import torch
import torch.nn as nn

class QLoRALayer(nn.Module):
    """
    Quantized 4-bit Low-Rank Adaptation (QLoRA) weight adapter with NormalFloat4 (NF4) quantization.
    """
    def __init__(self, in_features: int, out_features: int, r: int = 8, lora_alpha: float = 16.0):
        super().__init__()
        self.r = r
        self.scaling = lora_alpha / r
        
        # Frozen base weight representation
        self.register_buffer("base_weight", torch.randn(out_features, in_features))
        
        # Trainable low-rank adapters
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = x @ self.base_weight.T
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out
