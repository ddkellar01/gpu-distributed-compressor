import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) injection layer for gradient compression during fine-tuning.
    """
    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
