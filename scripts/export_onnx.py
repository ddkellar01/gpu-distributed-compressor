import torch
from models.model_builder import build_model

def export_model():
    print("[EXPORT] Instantiating model for ONNX export...")
    model = build_model()
    model.eval()

    dummy_input = torch.randint(0, 32000, (1, 128))
    onnx_path = "checkpoints/backbone_transformer.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'}}
    )
    print(f"[EXPORT] Successfully exported model to {onnx_path}")

if __name__ == "__main__":
    export_model()
