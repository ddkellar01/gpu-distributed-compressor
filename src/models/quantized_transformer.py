import torch
import torch.nn as nn

class INT8Linear(nn.Module):
    """
    Quantized linear layer for post-training gradient and weight quantization.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8), requires_grad=False)
        self.register_buffer('scale', torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.t().to(x.dtype)) * self.scale
