# env/allocation.py
from typing import Dict, Optional, Tuple

def greedy_allocate(
    gpu_state: Dict[str, int], 
    gpu_type_info: Dict[str, dict], 
    gpus_needed: int
) -> Optional[Tuple[Dict[str, int], Dict[str, int]]]:
    """
    Tries to fit the job onto a single GPU type first (preferring more powerful types),
    then splits across types if necessary.
    Returns (new_gpu_state, allocation_dict) on success, else None.
    """
    # Sort types by FLOPS (performance) to prefer better GPUs
    sorted_types = sorted(gpu_type_info.keys(), key=lambda t: gpu_type_info[t].get('flops', 0), reverse=True)

    # 1. Try to fit on a single best-performing type
    for gpu_type in sorted_types:
        if gpu_state.get(gpu_type, 0) >= gpus_needed:
            new_state = gpu_state.copy()
            new_state[gpu_type] -= gpus_needed
            allocation = {gpu_type: gpus_needed}
            return new_state, allocation

    # 2. If single-type fit fails, try to split across types (greedy)
    if sum(gpu_state.values()) >= gpus_needed:
        remaining_needed = gpus_needed
        new_state = gpu_state.copy()
        allocation = {}
        for gpu_type in sorted_types:
            available = new_state.get(gpu_type, 0)
            take = min(remaining_needed, available)
            if take > 0:
                allocation[gpu_type] = take
                new_state[gpu_type] -= take
                remaining_needed -= take
            if remaining_needed == 0:
                return new_state, allocation
    
    return None # Cannot satisfy the request

def homogeneous_allocate(
    gpu_state: Dict[str, int], 
    gpu_type_info: Dict[str, dict],
    gpus_needed: int
) -> Optional[Tuple[Dict[str, int], Dict[str, int]]]:
    """
    Only allocates if a single GPU type can satisfy the entire request.
    This prevents a job from being split across different hardware types.
    """
    sorted_types = sorted(gpu_type_info.keys(), key=lambda t: gpu_type_info[t].get('flops', 0), reverse=True)
    
    for gpu_type in sorted_types:
        if gpu_state.get(gpu_type, 0) >= gpus_needed:
            new_state = gpu_state.copy()
            new_state[gpu_type] -= gpus_needed
            allocation = {gpu_type: gpus_needed}
            return new_state, allocation
            
    return None