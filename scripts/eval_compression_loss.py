import torch
from models.model_builder import build_model
from compression.powersgd import PowerSGDCompressor
from compression.quantizer import Int8Quantizer

def evaluate_reconstruction_error():
    print("=== Compression Reconstruction Accuracy Evaluation ===")
    model = build_model()
    
    psgd = PowerSGDCompressor(rank=4)
    quant = Int8Quantizer(num_bits=8)

    for name, param in model.named_parameters():
        if param.requires_grad:
            dummy_grad = torch.randn_like(param)
            
            # PowerSGD evaluation
            psgd_comp, shape = psgd.compress(name, dummy_grad)
            psgd_decomp = psgd.decompress(psgd_comp, shape)
            psgd_err = torch.norm(dummy_grad - psgd_decomp) / torch.norm(dummy_grad)

            # INT8 evaluation
            q_data, scale = quant.quantize(dummy_grad)
            q_decomp = quant.decompress(q_data, scale)
            q_err = torch.norm(dummy_grad - q_decomp) / torch.norm(dummy_grad)

            print(f"Layer: {name:30s} | PowerSGD Rel Error: {psgd_err:.4f} | INT8 Rel Error: {q_err:.4f}")
            break

if __name__ == "__main__":
    evaluate_reconstruction_error()
