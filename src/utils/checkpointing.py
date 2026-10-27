import os
import torch

def save_checkpoint(model, optimizer, epoch: int, path: str = "checkpoints/model.pt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state, path)
    print(f"[CHECKPOINT] Saved checkpoint state to {path}")

def load_checkpoint(model, optimizer, path: str = "checkpoints/model.pt"):
    if not os.path.exists(path):
        print(f"[CHECKPOINT] No checkpoint found at {path}")
        return 0
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"[CHECKPOINT] Loaded checkpoint from {path} (Epoch {checkpoint['epoch']})")
    return checkpoint["epoch"]
