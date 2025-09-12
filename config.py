# config.py
import os
from typing import Dict, List

# -------------------
# Project paths
# -------------------
ROOT_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------
# Environment defaults
# -------------------
ENV = {
    "num_gpus": 56,
    "top_k": 16,
    "time_step": 1.0,
    "episode_max_time": 15_000_000,
    "max_queue_len": 200.0, # For observation normalization
    "job_arrival_mode": "synthetic", # "synthetic", "google", "alibaba"
    "trace_path": None, # Path to real-world trace data
}

# -------------------
# Synthetic workload generator params
# -------------------
WORKLOAD = {
    "arrival_lambda": 2.5,
    "runtime_mean": 125000.0,
    "runtime_std": 55000.0,
    "gpu_req_min": 1,
    "gpu_req_max": 8,
    "priority_levels": [0.5, 1.0, 2.5, 5.0],
    "flops_mean": 1e12,
    "flops_std": 5e11,
}

# -------------------
# RL training
# -------------------
TRAIN = {
    "seed": 0,
    "algo": "sac",  # Default algorithm is SAC
    "max_episodes": 200,
    "max_steps_per_episode": 50000,
    "save_interval": 10,
}

# -------------------
# D3QN Agent HParams
# -------------------
DQN = {
    "hidden_sizes": [256, 256],
    "lr": 1e-4,
    "gamma": 0.95,
    "batch_size": 128,
    "buffer_size": 100000,
    "target_update_freq": 500,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 20000,
}

# -------------------
# SAC Agent HParams
# -------------------
SAC = {
    "hidden_sizes": [256, 256],
    "lr_actor": 3e-4,
    "lr_critic": 3e-4,
    "lr_alpha": 3e-4,
    "gamma": 0.95,
    "tau": 0.005,      # Soft update coefficient
    "batch_size": 256,
    "buffer_size": 100000,
    "policy_delay": 2, # Delayed policy updates
    "target_entropy": "auto",
}

# -------------------
# Reward weights
# -------------------
REWARD = {
    "w_complete": 50.0, # Reward for completing a job (scaled by priority)
    "w_wait": 0.05,      # Penalty for job waiting time in the queue
    "w_util": 1.5,      # Reward for GPU utilization
    "w_flow": 0.3,     # Penalty for the number of jobs in the system
}

# -------------------
# GPU heterogeneity
# -------------------
GPU_TYPES = [
    {"name": "A100", "count": 16, "flops": 1.6e14},
    {"name": "V100", "count": 16, "flops": 7.8e13},
    {"name": "T4",   "count": 12, "flops": 8.1e12},
    {"name": "4090", "count": 12, "flops": 2.1e31},
]

# -------------------
# Allocation semantics
# -------------------
ALLOC_STRATEGY = "greedy"
RESERVATION_MODE = "direct"

# -------------------
# Cluster Config Builder
# -------------------
def make_cluster_cfg(*, gpu_types: List[Dict] = None, heterogeneous: bool = True) -> Dict:
    gpu_types = gpu_types if gpu_types is not None else GPU_TYPES
    cfg = {
        "heterogeneous": bool(heterogeneous),
        "RESERVATION_MODE": RESERVATION_MODE,
        "ALLOC_STRATEGY": ALLOC_STRATEGY,
        "gpu_type_info": {t["name"]: {"flops": t.get("flops", 0.0)} for t in gpu_types},
        "initial_gpu_state": {t["name"]: int(t.get("count", 0)) for t in gpu_types},
    }
    return cfg

DEFAULT_CLUSTER_CFG = make_cluster_cfg()
TOTAL_GPUS = sum(t.get("count", 0) for t in GPU_TYPES)
ENV["num_gpus"] = TOTAL_GPUS if TOTAL_GPUS > 0 else ENV["num_gpus"]