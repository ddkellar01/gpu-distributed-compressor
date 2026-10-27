import os

class WandbLogger:
    """
    Weights & Biases telemetry tracker for multi-GPU compression benchmarking.
    """
    def __init__(self, project_name: str = "gpu-distributed-compressor", enabled: bool = True):
        self.enabled = enabled and ("WANDB_API_KEY" in os.environ)
        if self.enabled:
            import wandb
            self.wandb = wandb
            self.wandb.init(project=project_name)

    def log(self, metrics: dict, step: int):
        if self.enabled:
            self.wandb.log(metrics, step=step)

    def finish(self):
        if self.enabled:
            self.wandb.finish()
