import time
import torch
from models.model_builder import build_model

def benchmark_throughput():
    print("[BENCHMARK] Starting training throughput benchmark...")
    model = build_model().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    batch_size = 16
    seq_len = 512
    inputs = torch.randint(0, 32000, (batch_size, seq_len), device="cuda")
    targets = torch.randint(0, 32000, (batch_size, seq_len), device="cuda")

    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        out = model(inputs)
        loss = torch.nn.functional.cross_entropy(out.view(-1, 32000), targets.view(-1))
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    start = time.perf_counter()
    steps = 50
    for _ in range(steps):
        optimizer.zero_grad()
        out = model(inputs)
        loss = torch.nn.functional.cross_entropy(out.view(-1, 32000), targets.view(-1))
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens_processed = steps * batch_size * seq_len
    throughput = tokens_processed / elapsed
    print(f"[BENCHMARK] Throughput: {throughput:.2f} tokens/sec")

if __name__ == "__main__":
    benchmark_throughput()
