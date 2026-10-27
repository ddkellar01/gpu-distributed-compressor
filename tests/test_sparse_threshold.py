import torch
from compression.lossless_huffman import SparseThresholdCompressor

def test_sparse_thresholding():
    compressor = SparseThresholdCompressor(threshold=0.5)
    data = torch.tensor([0.1, 0.8, -0.6, 0.05, 1.2])
    comp, shape = compressor.compress(data)
    decomp = compressor.decompress(comp)
    
    assert torch.all(decomp[data.abs() < 0.5] == 0)
    assert decomp.shape == shape
    print("[PASS] Sparse Threshold Compressor Test")

if __name__ == "__main__":
    test_sparse_thresholding()
