import time
import torch
import torch.distributed as dist
from utils.distributed import init_distributed

def benchmark_bandwidth():
    rank, local_rank, world_size = init_distributed()
    if world_size < 2:
        print("[BENCHMARK] Multi-GPU required to test NCCL interconnect bandwidth.")
        return

    tensor_size = 100 * 1024 * 1024  # 100M floats = 400 MB payload
    data = torch.randn(tensor_size, device=f"cuda:{local_rank}", dtype=torch.float32)

    torch.cuda.synchronize()
    start = time.perf_counter()
    dist.all_reduce(data)
    torch.cuda.synchronize()
    end = time.perf_counter()

    elapsed = end - start
    data_transferred_gb = 2 * (world_size - 1) / world_size * (tensor_size * 4) / 1e9
    bus_bandwidth = data_transferred_gb / elapsed

    if rank == 0:
        print(f"[BENCHMARK] Elapsed: {elapsed:.4f}s | Effective NCCL Bus Bandwidth: {bus_bandwidth:.2f} GB/s")

if __name__ == "__main__":
    benchmark_bandwidth()
