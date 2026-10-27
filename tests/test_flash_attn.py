import torch
from models.flash_attention_wrapper import FlashAttentionWrapper

def test_flash_attn():
    layer = FlashAttentionWrapper(embed_dim=256, num_heads=8)
    dummy_input = torch.randn(2, 64, 256)
    out = layer(dummy_input)
    assert out.shape == (2, 64, 256)
    print("[PASS] FlashAttention Wrapper Test")

if __name__ == "__main__":
    test_flash_attn()
