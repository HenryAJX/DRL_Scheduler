# metrics/logger.py
import pandas as pd
import os

def save_job_log(jobs, filename):
    rows = []
    for j in jobs:
        rows.append({
            "job_id": j.job_id,
            "user_id": j.user_id,
            "arrival_time": j.arrival_time,
            "start_time": j.start_time if j.start_time is not None else -1,
            "finish_time": j.finish_time if j.finish_time is not None else -1,
            "runtime": j.runtime,
            "gpus": j.gpus,
            "priority": j.priority,
            "flops": j.flops,
        })
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)

def append_summary(summary_row, filename):
    exists = os.path.exists(filename)
    df = pd.DataFrame([summary_row])
    # Simplified to remove config dependency
    df.to_csv(filename, mode='a', header=not exists, index=False)