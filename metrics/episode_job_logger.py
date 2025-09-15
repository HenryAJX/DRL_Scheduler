import os
import pandas as pd
from typing import List
from env.job import Job

def save_detailed_job_log(
    *,
    completed_jobs: List[Job],
    episode: int,
    log_dir: str,
    prefix: str
):
    """
    Saves a detailed log of all completed jobs for an episode by appending
    them to a single, run-specific CSV file.

    This creates one large file per training run, with an 'episode' column
    to distinguish data from different episodes.

    Args:
        completed_jobs: A list of Job objects that completed during the episode.
        episode: The current episode number.
        log_dir: The base directory for all logs.
        prefix: A unique prefix for the run (e.g., 'sac_seed0').
    """
    if not completed_jobs:
        return  # Nothing to log

    # Use a consistent filename for the entire run, stored in the base log directory
    filename = f"{prefix}_detailed_jobs.csv"
    log_path = os.path.join(log_dir, filename)

    rows = []
    for j in completed_jobs:
        if j.finish_time is None or j.start_time is None:
            continue
        
        rows.append({
            "episode": episode,  # Add episode number to the data
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

    df = pd.DataFrame(rows)
    
    # Check if the file already exists to decide whether to write the header
    header = not os.path.exists(log_path)
    
    # Append to the CSV file without creating a new directory
    df.to_csv(log_path, mode='a', header=header, index=False)

