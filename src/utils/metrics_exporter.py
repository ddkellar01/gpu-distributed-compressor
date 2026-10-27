import time
import torch
from prometheus_client import start_http_server, Gauge

GPU_MEMORY_GAUGE = Gauge('gpu_memory_allocated_mb', 'Allocated GPU Memory in MB', ['rank'])
COMPRESSION_RATIO_GAUGE = Gauge('compression_ratio', 'Current gradient compression ratio', ['layer'])

def start_metrics_server(port: int = 8000):
    start_http_server(port)
    print(f"[METRICS] Prometheus exporter listening on port {port}")

def log_gpu_metrics(rank: int):
    if torch.cuda.is_available():
        mem_mb = torch.cuda.memory_allocated(rank) / (1024 * 1024)
        GPU_MEMORY_GAUGE.labels(rank=str(rank)).set(mem_mb)
