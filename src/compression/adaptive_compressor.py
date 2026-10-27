import torch

class AdaptiveCompressor:
    """
    Dynamically adjusts compression ratios/ranks based on observed gradient norm and loss convergence rate.
    """
    def __init__(self, base_rank: int = 4, max_rank: int = 16, norm_threshold: float = 5.0):
        self.current_rank = base_rank
        self.max_rank = max_rank
        self.norm_threshold = norm_threshold

    def adjust_rank(self, grad_norm: float):
        if grad_norm > self.norm_threshold and self.current_rank < self.max_rank:
            self.current_rank = min(self.max_rank, self.current_rank * 2)
            print(f"[ADAPTIVE] High gradient variance ({grad_norm:.2f}). Scaling compression rank to {self.current_rank}")
        elif grad_norm < (self.norm_threshold / 2) and self.current_rank > 2:
            self.current_rank = max(2, self.current_rank // 2)
            print(f"[ADAPTIVE] Stable gradient norm ({grad_norm:.2f}). Reducing compression rank to {self.current_rank}")
        return self.current_rank
