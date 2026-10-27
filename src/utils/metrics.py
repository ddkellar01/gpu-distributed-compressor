import torch

def compute_compression_ratio(original_tensor: torch.Tensor, compressed_bytes: int) -> float:
    original_bytes = original_tensor.numel() * original_tensor.element_size()
    return original_bytes / max(1, compressed_bytes)

def compute_cosine_similarity(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> float:
    a_flat = tensor_a.view(-1).float()
    b_flat = tensor_b.view(-1).float()
    return (torch.dot(a_flat, b_flat) / (torch.norm(a_flat) * torch.norm(b_flat) + 1e-8)).item()
