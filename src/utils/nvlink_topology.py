import subprocess

def inspect_nvlink_topology():
    """
    Queries and parses nvidia-smi topo -m to verify GPU NVLink connectivity matrix.
    """
    try:
        output = subprocess.check_output(["nvidia-smi", "topo", "-m"]).decode("utf-8")
        print("=== GPU Interconnect Matrix ===")
        print(output)
        has_nvlink = "NV" in output
        print(f"[NVLINK] Direct NVLink Matrix Active: {has_nvlink}")
        return has_nvlink
    except Exception as e:
        print(f"[ERROR] Unable to query GPU topology: {e}")
        return False

if __name__ == "__main__":
    inspect_nvlink_topology()
