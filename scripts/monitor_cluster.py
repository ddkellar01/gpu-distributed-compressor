import time
import subprocess

def get_gpu_utilization():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
        ).decode("utf-8")
        
        print(f"\n--- GPU Cluster Metrics ({time.strftime('%H:%M:%S')}) ---")
        for line in output.strip().split("\n"):
            gpu_id, util, mem_used, mem_total = line.split(", ")
            print(f"GPU {gpu_id}: Util {util:>3}% | Memory {mem_used:>5} MB / {mem_total:>5} MB")
    except Exception as e:
        print(f"[ERROR] Could not query nvidia-smi: {e}")

if __name__ == "__main__":
    for _ in range(5):
        get_gpu_utilization()
        time.sleep(2)
