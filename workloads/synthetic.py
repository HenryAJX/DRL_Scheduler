# workloads/synthetic.py
import random
import math
from typing import List, Optional, Dict, Any

from env.job import Job
from config import WORKLOAD, GPU_TYPES


def _build_type_counts_from_cluster_cfg(cluster_cfg: Dict[str, Any]) -> Dict[str, int]:
    """
    Extract a mapping type->count from cluster_cfg which is expected to contain
    'initial_gpu_state' (dict) or 'gpu_type_info' + counts.
    """
    if not cluster_cfg:
        return {}
    if "initial_gpu_state" in cluster_cfg:
        return dict(cluster_cfg["initial_gpu_state"])
    # fallback: try GPU_TYPES-style list
    if "gpu_types" in cluster_cfg:
        return {t["name"]: int(t.get("count", 0)) for t in cluster_cfg["gpu_types"]}
    return {}


def _build_type_counts_from_gpu_types(gpu_types: List[Dict[str, Any]]) -> Dict[str, int]:
    """Convert GPU_TYPES list (from config) into a name->count dict."""
    return {t["name"]: int(t.get("count", 0)) for t in (gpu_types or [])}


def _greedy_allocation_hint(gpus_needed: int, type_counts: Dict[str, int]) -> Optional[Dict[str, int]]:
    """
    Produce a simple greedy allocation hint that tries to allocate from a single type
    if possible, otherwise splits across types preferring types with more available units.
    Returns None if total capacity < gpus_needed.
    """
    total = sum(type_counts.values())
    if total < gpus_needed:
        return None

    # Try single-type fit first: pick the type with largest count that can satisfy
    sorted_by_count = sorted(type_counts.items(), key=lambda kv: kv[1], reverse=True)
    for t, avail in sorted_by_count:
        if avail >= gpus_needed:
            return {t: gpus_needed}

    # Otherwise split greedily across types
    remaining = gpus_needed
    allocation = {}
    for t, avail in sorted_by_count:
        take = min(avail, remaining)
        if take > 0:
            allocation[t] = take
            remaining -= take
        if remaining == 0:
            break

    return allocation if remaining == 0 else None


def synthetic_workload_generator(
    max_time: float = 1000.0,
    arrival_lambda: Optional[float] = None,
    cluster_cfg: Optional[Dict[str, Any]] = None,
    **kwargs
) -> List[Job]:
    lam = arrival_lambda if arrival_lambda is not None else WORKLOAD.get("arrival_lambda", 0.5)
    if lam <= 0: return []

    t, job_id = 0.0, 0
    jobs: List[Job] = []
    users = [101, 102, 103]
    
    while t < max_time:
        t += -math.log(1.0 - random.random()) / lam
        if t > max_time: break

        runtime = max(1.0, random.gauss(WORKLOAD.get("runtime_mean"), WORKLOAD.get("runtime_std")))
        gpus = random.randint(WORKLOAD.get("gpu_req_min"), WORKLOAD.get("gpu_req_max"))
        
        job = Job(
            job_id=job_id,
            arrival_time=float(t),
            runtime=float(runtime),
            gpus=int(gpus),
            priority=random.choice(WORKLOAD.get("priority_levels", [1.0])),
            flops=max(1e9, random.gauss(WORKLOAD.get("flops_mean"), WORKLOAD.get("flops_std"))),
            user_id=random.choice(users)
        )
        jobs.append(job)
        job_id += 1

    jobs.sort(key=lambda x: x.arrival_time)
    return jobs
