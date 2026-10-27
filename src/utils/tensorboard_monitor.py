from torch.utils.tensorboard import SummaryWriter

class TensorboardMonitor:
    """
    Logs bandwidth savings, gradient norms, and training loss directly to TensorBoard traces.
    """
    def __init__(self, log_dir: str = "logs/tb_events"):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_metrics(self, step: int, loss: float, compression_ratio: float, bandwidth_gbps: float):
        self.writer.add_scalar("Loss/train", loss, step)
        self.writer.add_scalar("Compression/Ratio", compression_ratio, step)
        self.writer.add_scalar("Network/Bandwidth_GBps", bandwidth_gbps, step)

    def close(self):
        self.writer.close()
