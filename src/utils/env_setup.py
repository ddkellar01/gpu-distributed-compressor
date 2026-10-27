import os
import torch

def setup_environment(seed: int = 42):
    """
    Set deterministic seeds and environment configs across processes.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[ENV] Set global random seed to {seed}")
