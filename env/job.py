# env/job.py
from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass
class Job:
    job_id: int
    user_id: int
    arrival_time: float
    runtime: float
    gpus: int
    priority: float
    flops: float
    
    # These attributes are populated by the environment at runtime
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    allocation: Optional[Dict[str, int]] = None