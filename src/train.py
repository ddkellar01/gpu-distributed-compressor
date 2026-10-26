import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from compression.powersgd import PowerSGDCommunicator
from models.model_builder import create_model
from utils.distributed import init_distributed_env
from utils.lr_scheduler import WarmupCosineAnnealingLR

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    return parser.parse_args()

def main():
    args = parse_args()

    local_rank, rank, world_size = init_distributed_env(backend="nccl")
    device = torch.device("cuda", local_rank)

    # Build model and wrap DDP
    model = create_model(hidden_dim=2048).to(device)
    model = DDP(model, device_ids=[local_rank])
    
    # Register PowerSGD Hook
    powersgd = PowerSGDCommunicator(rank_matrix=4)
    
    # Synthetic Data & Sampler
    x_data = torch.randn(1000, 2048)
    y_data = torch.randn(1000, 2048)
    dataset = TensorDataset(x_data, y_data)
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(loader) * args.epochs // args.grad_accum_steps
    scheduler = WarmupCosineAnnealingLR(optimizer, warmup_steps=int(total_steps * 0.1), max_steps=total_steps)
    criterion = nn.MSELoss()

    model.train()
    step_count = 0

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        optimizer.zero_grad()

        for step, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets) / args.grad_accum_steps
            loss.backward()

            if (step + 1) % args.grad_accum_steps == 0:
                for param in model.parameters():
                    if param.grad is not None:
                        powersgd.compress_and_allreduce(param.grad)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step_count += 1

                if local_rank == 0 and step_count % 10 == 0:
                    print(f"Epoch: {epoch} | Step: {step_count} | Loss: {loss.item() * args.grad_accum_steps:.4f} | LR: {scheduler.get_lr()[0]:.6f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
