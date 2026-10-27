import torch

class ZeroBubbleScheduler:
    """
    Schedules forward and backward micro-batches to eliminate pipeline bubbles in distributed training.
    """
    def __init__(self, num_stages: int = 4, num_micro_batches: int = 16):
        self.num_stages = num_stages
        self.num_micro_batches = num_micro_batches

    def get_schedule(self):
        schedule = []
        for mb in range(self.num_micro_batches):
            schedule.append(("F", mb))
        for mb in range(self.num_micro_batches):
            schedule.append(("B", mb))
        return schedule
