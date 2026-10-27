import torch
import torch.nn as nn

class MambaSSMBlock(nn.Module):
    """
    Selective State Space Model (SSM) block integrated into the distributed model constructor.
    """
    def __init__(self, d_model: int = 1024, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape
        x_proj = self.in_proj(x).chunk(2, dim=-1)[0]
        x_conv = self.conv1d(x_proj.transpose(1, 2))[:, :, :s].transpose(1, 2)
        return self.out_proj(torch.silu(x_conv))
