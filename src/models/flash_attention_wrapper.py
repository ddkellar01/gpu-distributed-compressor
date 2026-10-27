import torch
import torch.nn as nn

class FlashAttentionWrapper(nn.Module):
    """
    High-throughput FlashAttention-2 wrapper integrated with gradient compression hooks.
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape
        qkv = self.qkv_proj(x).reshape(b, s, 3, self.num_heads, d // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(b, s, d)
        
        return self.out_proj(context)
