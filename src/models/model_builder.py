import torch.nn as nn

def create_model(hidden_dim=2048):
    """
    Builds a baseline neural network model backbone for cluster scaling and benchmarks.
    """
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim * 2),
        nn.GELU(),
        nn.LayerNorm(hidden_dim * 2),
        nn.Linear(hidden_dim * 2, hidden_dim)
    )
