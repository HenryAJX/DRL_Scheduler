# In metrics/episode_job_logger.py

import os
import pandas as pd
from typing import List
# Note: You may need to adjust the import path for Job depending on your project structure
from env.job import Job 

def save_detailed_job_log(completed_jobs: List[Job], log_path: str):
    """
    Saves a detailed log of completed jobs to a specific CSV file, overwriting
    the file if it already exists. This is suitable for evaluation and partial saves.

    Args:
        completed_jobs: A list of all Job objects completed so far.
        log_path: The full, direct path to the output CSV file.
    """
    if not completed_jobs:
        return  # Nothing to log

    rows = []
    for j in completed_jobs:
        if j.finish_time is None or j.start_time is None:
            continue
        
        rows.append({
            "job_id": j.job_id,
            "user_id": j.user_id,
            "priority": j.priority,
            "gpus": j.gpus,
            "runtime": j.runtime,
            "arrival_time": j.arrival_time,
            "start_time": j.start_time,
            "finish_time": j.finish_time,
            "jct": j.finish_time - j.arrival_time,
            "wait_time": j.start_time - j.arrival_time,
            "flops": j.flops
        })
    
    if not rows:
        return

    try:
        df = pd.DataFrame(rows)
        # Write to the CSV, overwriting the file completely
        df.to_csv(log_path, mode='w', header=True, index=False)
    except Exception as e:
        print(f"[Logger Error] Failed to save detailed job log to {log_path}: {e}")