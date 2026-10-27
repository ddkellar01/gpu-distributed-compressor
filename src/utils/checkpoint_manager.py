import os
import glob
import torch

class CheckpointManager:
    """
    Manages rolling checkpoints with retention policies and metadata tracking.
    """
    def __init__(self, checkpoint_dir: str = "checkpoints", max_to_keep: int = 5):
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(self, state_dict: dict, step: int, score: float = 0.0):
        path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{step}_score_{score:.4f}.pt")
        torch.save(state_dict, path)
        print(f"[CHECKPOINT] Saved checkpoint: {path}")
        self._rotate_checkpoints()

    def _rotate_checkpoints(self):
        pattern = os.path.join(self.checkpoint_dir, "checkpoint_step_*.pt")
        files = sorted(glob.glob(pattern), key=os.path.getmtime)
        while len(files) > self.max_to_keep:
            oldest = files.pop(0)
            os.remove(oldest)
            print(f"[CHECKPOINT] Pruned old checkpoint: {oldest}")
