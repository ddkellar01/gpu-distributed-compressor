import torch
from torch.utils.data import Dataset, DataLoader

class SyntheticDataset(Dataset):
    """
    High-throughput synthetic dataset generator for benchmarking gradient compression algorithms.
    """
    def __init__(self, vocab_size: int = 32000, seq_len: int = 512, length: int = 10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        y = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        return x, y

def get_dataloader(batch_size: int = 16, num_workers: int = 4):
    dataset = SyntheticDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
