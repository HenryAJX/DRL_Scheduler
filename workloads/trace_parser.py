# workloads/trace_parser.py
import pandas as pd
from typing import List
from env.job import Job

def parse_workload_trace(file_path: str, trace_type: str) -> List[Job]:
    """
    Parses a workload trace file (CSV) and converts it into a list of Job objects.
    """
    df = pd.read_csv(file_path)
    jobs = []
    
    if trace_type == 'google':
        # Example for a simplified Google trace format
        # Timestamps are often in microseconds, need to convert to seconds
        df = df.rename(columns={
            'submit_time': 'arrival_time',
            'finish_time': 'end_time',
            'scheduling_class': 'priority',
            'resource_request_for_1_task_in_1.0_-_CPU_and_memory': 'gpus' # Placeholder mapping
        })
        # Normalize timestamps (assuming start from 0)
        min_time = df['arrival_time'].min()
        df['arrival_time'] = (df['arrival_time'] - min_time) / 1_000_000.0
        df['end_time'] = (df['end_time'] - min_time) / 1_000_000.0
        df['runtime'] = df['end_time'] - df['arrival_time']
        
        # Filter out failed or very short jobs
        df = df[df['runtime'] > 1.0]

        for i, row in df.iterrows():
            jobs.append(Job(
                job_id=i,
                arrival_time=row['arrival_time'],
                runtime=row['runtime'],
                gpus=int(row.get('gpus', 1)), # Default to 1 if not specified
                priority=float(row.get('priority', 1.0)),
                flops=row['runtime'] * 1e12, # Synthesize FLOPS based on runtime
                user_id=0

            ))
            
    elif trace_type == 'alibaba':
        # Example for a simplified Alibaba trace format
        df = df.rename(columns={
            'start_time': 'arrival_time',
            'end_time': 'end_time',
            'instance_num': 'gpus',
        })
        min_time = df['arrival_time'].min()
        df['arrival_time'] = df['arrival_time'] - min_time
        df['end_time'] = df['end_time'] - min_time
        df['runtime'] = df['end_time'] - df['arrival_time']

        df = df[df['runtime'] > 1.0]

        for i, row in df.iterrows():
             jobs.append(Job(
                job_id=i,
                arrival_time=row['arrival_time'],
                runtime=row['runtime'],
                gpus=int(row.get('gpus', 1)),
                priority=1.0, # Default priority
                flops=row['runtime'] * 1e12,
                user_id=0
            ))
    else:
        raise ValueError(f"Unknown trace type: {trace_type}")
        
    # Sort jobs by arrival time, which is crucial for the simulation
    jobs.sort(key=lambda j: j.arrival_time)
    return jobs