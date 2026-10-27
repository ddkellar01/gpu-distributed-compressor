import torch
from compression.qlora import QLoRALayer

def test_qlora_layer():
    layer = QLoRALayer(in_features=512, out_features=1024, r=16)
    x = torch.randn(4, 512)
    output = layer(x)
    assert output.shape == (4, 1024)
    print("[PASS] QLoRA Layer Test")

if __name__ == "__main__":
    test_qlora_layer()
