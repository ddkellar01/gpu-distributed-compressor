import torch
from safetensors.torch import save_file, load_file
from models.model_builder import build_model

def convert_to_safetensors(pt_path: str, st_path: str):
    print(f"[CONVERT] Converting PyTorch checkpoint {pt_path} to SafeTensors...")
    state_dict = torch.load(pt_path, map_location="cpu")
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
        
    save_file(state_dict, st_path)
    print(f"[CONVERT] Saved SafeTensors file to {st_path}")

if __name__ == "__main__":
    model = build_model()
    torch.save(model.state_dict(), "checkpoints/temp.pt")
    convert_to_safetensors("checkpoints/temp.pt", "checkpoints/model.safetensors")
