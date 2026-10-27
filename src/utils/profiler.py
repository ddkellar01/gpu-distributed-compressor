import torch
from contextlib import contextmanager

@contextmanager
def torch_profile_scope(output_path: str = "logs/trace.json"):
    """
    Context manager for PyTorch profiler and chrome trace export.
    """
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("logs/tensorboard"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        yield prof
    prof.export_chrome_trace(output_path)
    print(f"[PROFILER] Trace exported to {output_path}")
