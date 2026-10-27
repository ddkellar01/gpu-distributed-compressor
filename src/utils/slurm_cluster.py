import os
import subprocess

def get_slurm_environment():
    """
    Auto-detects SLURM execution environment variables for multi-node setup.
    """
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if not slurm_job_id:
        return None

    nodelist = os.environ.get("SLURM_JOB_NODELIST")
    master_node = subprocess.check_output(["scontrol", "show", "hostnames", nodelist]).decode().splitlines()[0]
    
    return {
        "job_id": slurm_job_id,
        "master_node": master_node,
        "nnodes": int(os.environ.get("SLURM_JOB_NUM_NODES", 1)),
        "node_rank": int(os.environ.get("SLURM_NODEID", 0)),
        "gpus_per_node": int(os.environ.get("SLURM_GPUS_PER_NODE", 8))
    }
