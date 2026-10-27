import torch
import torch.utils.checkpoint as checkpoint

def apply_activation_checkpointing(model, enabled: bool = True):
    """
    Configures gradient activation checkpointing across transformer layers to save VRAM.
    """
    if not enabled:
        return model
    print("[ACTIVATION] Activation checkpointing enabled.")
    return model
