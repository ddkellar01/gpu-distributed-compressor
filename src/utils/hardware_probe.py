import torch
import psutil

def inspect_hardware_environment():
    """
    Inspects host CPU, RAM, and GPU capabilities.
    """
    info = {
        "cpu_count": psutil.cpu_count(logical=True),
        "system_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    return info

if __name__ == "__main__":
    print(inspect_hardware_environment())
