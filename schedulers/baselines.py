# schedulers/baselines.py
from typing import Any, List, Optional
from env.job import Job

def _available_total(gpu_state: Any) -> int:
    """Return total available GPU units from the state dictionary."""
    if isinstance(gpu_state, dict):
        return sum(int(v) for v in gpu_state.values())
    return 0

def _fits_in_any_type(required: int, gpu_state: Any) -> bool:
    """For heterogeneous dicts, check if any single GPU type has enough units."""
    if isinstance(gpu_state, dict):
        for avail in gpu_state.values():
            if int(avail) >= required:
                return True
        return False
    return False

def fcfs_choice(pending_queue: List[Any], gpu_state: Any) -> Optional[int]:
    """First-Come-First-Served: pick the first job in the queue that fits."""
    for idx, job_tuple in enumerate(pending_queue):
        job = job_tuple[3]
        req = job.gpus
        if _fits_in_any_type(req, gpu_state):
            return idx
    return None

def sjf_choice(pending_queue: List[Any], gpu_state: Any) -> Optional[int]:
    """Shortest-Job-First among jobs that fit."""
    best_idx = None
    best_runtime = float("inf")
    for idx, job_tuple in enumerate(pending_queue):
        job = job_tuple[3]
        req = job.gpus
        if not _fits_in_any_type(req, gpu_state):
            continue
        
        runtime = job.runtime
        if runtime < best_runtime:
            best_runtime = runtime
            best_idx = idx
    return best_idx

def first_fit_choice(pending_queue: List[Any], gpu_state: Any) -> Optional[int]:
    """Same as FCFS for our purposes."""
    return fcfs_choice(pending_queue, gpu_state)