import torch
from compression.powersgd import PowerSGDCompressor
from compression.quantizer import Int8Quantizer

def test_powersgd():
    compressor = PowerSGDCompressor(rank=2)
    grad = torch.randn(128, 128)
    comp, shape = compressor.compress("layer1", grad)
    decomp = compressor.decompress(comp, shape)
    assert decomp.shape == grad.shape
    print("[PASS] PowerSGD Unit Test")

def test_int8_quantizer():
    quantizer = Int8Quantizer(num_bits=8)
    grad = torch.randn(256, 256)
    q, scale = quantizer.quantize(grad)
    decomp = quantizer.decompress(q, scale)
    assert decomp.shape == grad.shape
    print("[PASS] Int8Quantizer Unit Test")

if __name__ == "__main__":
    test_powersgd()
    test_int8_quantizer()
