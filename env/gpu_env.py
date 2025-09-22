# env/gpu_env.py
import bisect
import random
from typing import Optional, Tuple, List, Dict

import simpy
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.job import Job
from config import ENV, WORKLOAD, REWARD, DEFAULT_CLUSTER_CFG
from env.allocation import greedy_allocate, homogeneous_allocate
from workloads.synthetic import synthetic_workload_generator

class GPUClusterEnv(gym.Env):
    """
    SimPy-backed GPU cluster environment, compatible with Gymnasium.
    Supports both homogeneous and heterogeneous cluster configurations
    with a unified dictionary-based state representation.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, config: Optional[dict] = None, cluster_cfg: Optional[dict] = None):
        config = config or ENV
        self.top_k = config.get("top_k", 8)
        self.time_step = config.get("time_step", 1.0)
        self.episode_max_time = config.get("episode_max_time", 2000)
        self.max_queue_len = config.get("max_queue_len", 100.0)

        # Reward weights
        self.w_complete = REWARD["w_complete"]
        self.w_wait = REWARD["w_wait"]
        self.w_util = REWARD["w_util"]
        self.w_flow = REWARD["w_flow"]

        # Cluster configuration
        self.cluster_cfg = cluster_cfg or DEFAULT_CLUSTER_CFG
        self.heterogeneous = self.cluster_cfg.get("heterogeneous", False)
        self.initial_gpu_state = self.cluster_cfg["initial_gpu_state"]
        self.num_gpus = sum(self.initial_gpu_state.values())

        # SimPy environment placeholders
        self.env: Optional[simpy.Environment] = None
        self.gpu_state: Optional[Dict[str, int]] = None

        # Bookkeeping
        self.pending_queue: List[Job] = []
        self.running_jobs: Dict[int, Job] = {}
        self.completed_jobs: List[Job] = []
        self.workload: List[Job] = []
        self.next_job_index = 0
        self.now = 0.0

        # Observation/action spaces
        obs_dim = 1 + self.top_k * 4 + 1  # gpu_avail_frac, k*job_features, queue_len_norm
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.top_k + 1) # top_k jobs + 1 no-op action

        # Feature normalizers
        self._max_runtime = WORKLOAD["runtime_mean"] * 5
        self._max_flops = WORKLOAD["flops_mean"] * 5
        self._max_gpus = max(1, self.num_gpus)
        self._max_priority = max(WORKLOAD.get("priority_levels", [1.0]))
        

    def reset(self, *, seed: Optional[int] = None, workload: Optional[list] = None, options=None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.env = simpy.Environment()
        self.gpu_state = dict(self.initial_gpu_state)

        # Reset bookkeeping
        self.pending_queue = []
        self.running_jobs = {}
        self.completed_jobs = []
        self.workload = workload if workload is not None else synthetic_workload_generator(self.episode_max_time)
        self.next_job_index = 0
        self.now = 0.0

        self.env.process(self._job_arrival_process())
        
        obs = self._get_observation()
        return obs.astype(np.float32), {}

    def step(self, action: int):
        assert self.env is not None
        info = {}
        reward = 0.0
        
        # Case 1: The action is a valid index for a pending job
        if 0 <= action < self.top_k and action < len(self.pending_queue):
            job_to_schedule = self.pending_queue[action][3]
            
            alloc_func = greedy_allocate if self.cluster_cfg["ALLOC_STRATEGY"] == "greedy" else homogeneous_allocate
            result = alloc_func(self.gpu_state, self.cluster_cfg["gpu_type_info"], job_to_schedule.gpus)

            if result:
                new_state, allocation = result
                self.gpu_state = new_state
                job_to_schedule.allocation = allocation
                self.env.process(self._run_job(job_to_schedule))
                self.running_jobs[job_to_schedule.job_id] = job_to_schedule
                self.pending_queue.pop(action)
        # Case 2: The action is the EXPLICIT "do nothing" action
        elif action == self.top_k:
            pass
        # Case 3: The action is invalid (out of bounds for the current state)
        else:
            reward -= 0.1

        # Advance simulation time
        prev_completed_count = len(self.completed_jobs)
        self.env.run(until=self.now + self.time_step)
        self.now = self.env.now

        # Compute reward components
        newly_completed_jobs = self.completed_jobs[prev_completed_count:]
        
        # 1. Throughput & Priority Reward
        reward_complete = self.w_complete * sum(j.priority for j in newly_completed_jobs)
        
        # 2. Waiting Time Penalty
        penalty_wait = self.w_wait * sum((self.now - job_tuple[1]) / 1000.0 for job_tuple in self.pending_queue)
        
        # 3. Utilization Reward
        idle_gpus = sum(self.gpu_state.values())
        reward_util = self.w_util * (self.num_gpus - idle_gpus) / self.num_gpus
        
        # 4. System Flow Penalty
        penalty_flow = self.w_flow * (len(self.pending_queue) + len(self.running_jobs))
        
        reward += reward_complete - penalty_wait + reward_util - penalty_flow

        # Get next state and termination conditions
        obs = self._get_observation()
        terminated = self.next_job_index >= len(self.workload) and not self.pending_queue and not self.running_jobs
        truncated = self.now >= self.episode_max_time
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def _job_arrival_process(self):
        """A SimPy process that adds jobs to the pending queue based on their arrival times."""
        while self.next_job_index < len(self.workload):
            job = self.workload[self.next_job_index]
            # Wait until the job's arrival time
            if job.arrival_time > self.env.now:
                yield self.env.timeout(job.arrival_time - self.env.now)
            
            # Create a tuple that sorts correctly: high priority (negative makes it descending), then low arrival time.
            job_tuple = (-job.priority, job.arrival_time, job.job_id, job)

            # Insert the tuple efficiently while maintaining the sorted order.
            bisect.insort(self.pending_queue, job_tuple)

            self.next_job_index += 1


    def _run_job(self, job: Job):
        """A SimPy process that simulates a job's execution and handles its completion."""
        job.start_time = self.env.now
        try:
            yield self.env.timeout(job.runtime)
        finally:
            self._finalize_job(job)

    def _finalize_job(self, job: Job):
        """Helper to release resources and move a job to the completed list."""
        job.finish_time = self.env.now
        
        # Release GPU allocation
        if job.allocation:
            for gpu_type, count in job.allocation.items():
                self.gpu_state[gpu_type] += count
        
        self.completed_jobs.append(job)
        self.running_jobs.pop(job.job_id, None)

    def _get_observation(self) -> np.ndarray:
        # GPU availability feature
        total_avail = sum(self.gpu_state.values())
        gpu_avail_frac = float(total_avail) / float(self.num_gpus) if self.num_gpus > 0 else 0.0

        # --- OPTIMIZATION ---
        # Only consider a slice of the queue, not the whole thing
        q_len = int(min(len(self.pending_queue), self.max_queue_len))

        # Job features
        obs_jobs = []
        for i in range(self.top_k):
            if i < q_len:
                j = self.pending_queue[i][3]
                # Normalize features to be roughly in [-1, 1]
                p = float(j.priority) / self._max_priority if self._max_priority > 0 else 0.0
                fl = float(j.flops) / self._max_flops
                g = float(j.gpus) / self._max_gpus
                rt = float(min(j.runtime, self._max_runtime)) / self._max_runtime
                obs_jobs.extend([p, fl, g, rt])
            else:
                obs_jobs.extend([0.0, 0.0, 0.0, 0.0])
        
        # Queue length feature
        queue_len_norm = float(len(self.pending_queue)) / self.max_queue_len
        
        obs = np.array([gpu_avail_frac] + obs_jobs + [queue_len_norm], dtype=np.float32)
        return obs