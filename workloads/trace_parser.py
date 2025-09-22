import pandas as pd
from typing import List
from env.job import Job

def parse_workload_trace(file_path: str, trace_type: str) -> List[Job]:
    """
    Parses a workload trace file (CSV) and converts it into a list of Job objects.
    """
    df = pd.read_csv(file_path)
    jobs = []
    
    # This parser now expects a pre-cleaned file for Alibaba traces
    if trace_type in ['google', 'alibaba']:
        # The cleaning script standardizes the column names to what we need.
        # This logic now works for both Google and our cleaned Alibaba trace.
        
        # Ensure correct data types
        df['arrival_time'] = pd.to_numeric(df['arrival_time'])
        df['runtime'] = pd.to_numeric(df['runtime'])
        df['gpus'] = pd.to_numeric(df.get('gpus', 1)).astype(int)

        # Filter out failed or very short jobs again just in case
        df = df[df['runtime'] > 1.0]

        for i, row in df.iterrows():
            jobs.append(Job(
                job_id=i,
                arrival_time=row['arrival_time'],
                runtime=row['runtime'],
                gpus=int(row.get('gpus', 1)),
                # --- THIS LINE IS THE KEY ---
                # Read the priority from the preprocessed file, with a safe default.
                priority=float(row.get('priority', 1.0)),
                flops=row['runtime'] * int(row.get('gpus', 1)) * 1e12,
                user_id=0
            ))
            
    else:
        raise ValueError(f"Unknown trace type: {trace_type}")
        
    # Sort jobs by arrival time, which is crucial for the simulation
    jobs.sort(key=lambda j: j.arrival_time)
    return jobs
