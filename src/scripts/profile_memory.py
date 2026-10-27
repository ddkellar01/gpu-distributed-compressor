import torch
from models.model_builder import build_model

def profile_memory():
    print("=== GPU Memory Profiler ===")
    if not torch.cuda.is_available():
        print("CUDA device not available.")
        return

    torch.cuda.reset_peak_memory_stats()
    model = build_model().cuda()
    inputs = torch.randint(0, 32000, (8, 512)).cuda()
    
    alloc_mem = torch.cuda.memory_allocated() / 1e6
    print(f"[MEMORY] Model Parameter Allocation: {alloc_mem:.2f} MB")

    outputs = model(inputs)
    loss = outputs.sum()
    loss.backward()

    peak_mem = torch.cuda.max_memory_allocated() / 1e6
    print(f"[MEMORY] Peak Forward + Backward Allocation: {peak_mem:.2f} MB")

if __name__ == "__main__":
    profile_memory()
