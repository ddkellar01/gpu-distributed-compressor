import os
import signal
import sys
from utils.checkpointing import save_checkpoint

class GracefulExitHandler:
    """
    Catches termination signals (SIGTERM, SIGINT) in distributed clusters to save intermediate state.
    """
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        print(f"\n[FAULT TOLERANCE] Received signal {signum}. Saving emergency checkpoint...")
        save_checkpoint(self.model, self.optimizer, epoch=-1, path="checkpoints/emergency_save.pt")
        sys.exit(0)
