import torch
import torch.nn as nn

class MixtureOfExperts(nn.Module):
    """
    Distributed Mixture of Experts (MoE) layer with top-2 router gating for high parameter throughput.
    """
    def __init__(self, num_experts: int = 8, d_model: int = 1024, d_ff: int = 4096):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape
        x_flat = x.view(-1, d)
        logits = self.gate(x_flat)
        weights, indices = torch.topk(torch.softmax(logits, dim=-1), 2, dim=-1)

        output = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            mask = (indices == i).any(dim=-1)
            if mask.any():
                output[mask] += self.experts[i](x_flat[mask])

        return output.view(b, s, d)
