import torch
from compression.adaptive_compressor import AdaptiveCompressor

def test_adaptive_scaling():
    adaptive = AdaptiveCompressor(base_rank=4, max_rank=16, norm_threshold=5.0)
    
    # Test high norm increase
    new_rank = adaptive.adjust_rank(10.0)
    assert new_rank == 8
    
    # Test low norm decrease
    new_rank = adaptive.adjust_rank(1.0)
    assert new_rank == 4
    print("[PASS] Adaptive Compressor Test")

if __name__ == "__main__":
    test_adaptive_scaling()
