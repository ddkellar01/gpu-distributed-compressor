import torch
from compression.chunked_quantizer import ChunkedQuantizer

def test_chunked_quantization():
    quantizer = ChunkedQuantizer(chunk_size=1024, num_bits=8)
    data = torch.randn(4096)
    comp, shape = quantizer.quantize_chunked(data)
    decomp = quantizer.decompress_chunked(comp)
    assert decomp.shape == shape
    print("[PASS] Chunked Quantizer Test")

if __name__ == "__main__":
    test_chunked_quantization()
