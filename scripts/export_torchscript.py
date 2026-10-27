import torch
from models.model_builder import build_model

def export_torchscript():
    print("[EXPORT] Tracing model with TorchScript...")
    model = build_model()
    model.eval()

    example_input = torch.randint(0, 32000, (1, 128))
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save("checkpoints/backbone_transformer.ptc")
    print("[EXPORT] Saved TorchScript traced model to checkpoints/backbone_transformer.ptc")

if __name__ == "__main__":
    export_torchscript()
