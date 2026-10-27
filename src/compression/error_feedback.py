import torch

class ErrorFeedbackBuffer:
    """
    Error feedback (EF21 / EF-SGD) accumulator to preserve residual compression noise across steps.
    """
    def __init__(self):
        self.residuals = {}

    def compensate(self, name: str, grad: torch.Tensor) -> torch.Tensor:
        if name not in self.residuals:
            self.residuals[name] = torch.zeros_like(grad)
        return grad + self.residuals[name]

    def update_residual(self, name: str, original_grad: torch.Tensor, decompressed_grad: torch.Tensor):
        self.residuals[name] = original_grad - decompressed_grad
