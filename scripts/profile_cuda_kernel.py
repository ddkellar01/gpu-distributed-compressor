import torch
from compression.powersgd import PowerSGDCompressor

def profile_kernel_execution():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA device required for kernel profiling.")
        return

    compressor = PowerSGDCompressor(rank=8)
    grad = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    comp_data, shape = compressor.compress("profile_layer", grad)
    decomp = compressor.decompress(comp_data, shape)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"[PROFILER] Compression + Decompression Latency: {elapsed_time_ms:.3f} ms")

if __name__ == "__main__":
    profile_kernel_execution()
