import math
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineAnnealingLR(_LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by cosine annealing decay.
    """
    def __init__(self, optimizer, warmup_steps, max_steps, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            return [base_lr * (step / max(1, self.warmup_steps)) for base_lr in self.base_lrs]
        
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return [
            self.min_lr + (base_lr - self.min_lr) * cosine_decay
            for base_lr in self.base_lrs
        ]
