from tqdm import tqdm
import argparse
import os
import random
import numpy as np
import torch
import math
import json
import sys
from datetime import datetime

from env.gpu_env import GPUClusterEnv
from schedulers.baselines import fcfs_choice, sjf_choice, first_fit_choice
from agents.d3qn import D3QNAgent
from agents.sac import SACAgent

from workloads.synthetic import synthetic_workload_generator
from workloads.trace_parser import parse_workload_trace
from metrics.logger import save_job_log, append_summary
from metrics.plots import (
    jct_vs_arrival_plot,
    per_priority_jct_plots,
    flops_weighted_utilization_plot,
)
# --- CHANGE: Import the new detailed logger ---
from metrics.episode_job_logger import save_detailed_job_log
from config import TRAIN, DQN, SAC, MODEL_DIR, LOG_DIR

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")


# ------------------------
# Utility / checkpoint helpers
# ------------------------

def compute_avg_jct(jobs):
    if not jobs:
        return 0.0
    return float(np.mean([j.finish_time - j.arrival_time for j in jobs if j.finish_time is not None]))


def compute_utilization(env, makespan):
    if makespan <= 0:
        return 0.0
    busy_time = sum([(j.finish_time - j.start_time) * j.gpus for j in env.completed_jobs if j.start_time is not None and j.finish_time is not None])
    return busy_time / (env.num_gpus * makespan)


def _save_eval_outputs(env, prefix):
    # save job logs
    fname = os.path.join(LOG_DIR, f"{prefix}_jobs.csv")
    save_job_log(env.completed_jobs, fname)

    # compute metrics
    completed_jobs = [j for j in env.completed_jobs if j.finish_time is not None]
    avg_jct = compute_avg_jct(completed_jobs)
    makespan = max([j.finish_time for j in completed_jobs]) if completed_jobs else 0.0
    utilization = compute_utilization(env, makespan) if makespan > 0 else 0.0

    # --- ADDED: SLOWDOWN CALCULATION ---
    slowdowns = []
    for job in completed_jobs:
        # Assumes the job object has a 'runtime' attribute for its actual execution time
        # This check prevents division by zero for very short or invalid jobs.
        if hasattr(job, 'runtime') and job.runtime > 1e-6:
            time_in_system = job.finish_time - job.arrival_time
            slowdown = time_in_system / job.runtime
            slowdowns.append(slowdown)

    if not slowdowns:
        avg_slowdown = -1.0
        max_slowdown = -1.0
    else:
        avg_slowdown = np.mean(slowdowns)
        max_slowdown = np.max(slowdowns)
    # --- END OF ADDITION ---

    summary = {
        "algo": prefix,
        "seed": 0,
        "episode": 0,
        "avg_jct": avg_jct,
        "makespan": makespan,
        "utilization": utilization,
        "num_jobs": len(completed_jobs),
        "avg_slowdown": avg_slowdown,
        "max_slowdown": max_slowdown,
    }
    append_summary(summary, os.path.join(LOG_DIR, f"summary_eval.csv"))
    
    # plots
    outdir = LOG_DIR
    jct_vs_arrival_plot(completed_jobs, outdir, prefix=prefix)
    per_priority_jct_plots(completed_jobs, outdir, prefix=prefix)
    flops_weighted_utilization_plot(completed_jobs, env, outdir, prefix=prefix)
    print(f"[eval] Evaluation outputs saved under {LOG_DIR} with prefix {prefix}")


# ------------------------
# Checkpoint / metadata helpers
# ------------------------

def _meta_path_for(model_path: str) -> str:
    return model_path + ".meta.json"


def write_checkpoint_meta(model_path: str, last_completed_episode: int, seed: int):
    meta = {
        "last_completed_episode": int(last_completed_episode),
        "seed": int(seed),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    try:
        with open(_meta_path_for(model_path), "w") as f:
            json.dump(meta, f)
    except Exception as e:
        print(f"[meta] Failed to write metadata for {model_path}:", e)


def read_checkpoint_meta(model_path: str):
    meta_path = _meta_path_for(model_path)
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[meta] Failed to read metadata for {model_path}:", e)
        return None


def save_model_with_meta(agent, model_path: str, last_completed_episode: int, seed: int):
    try:
        agent.save(model_path)
        write_checkpoint_meta(model_path, last_completed_episode, seed)
        print(f"[ckpt] Saved model and metadata to {model_path}")
    except Exception as e:
        print(f"[ckpt] Failed to save model {model_path}:", e)


# ------------------------
# Baselines
# ------------------------

def run_baseline(algo, seed, workload):
    env = GPUClusterEnv()
    obs, _ = env.reset(seed=seed, workload=workload)
    terminated = truncated = False
    
    # --- NEW: Parameters for partial saving ---
    log_path = os.path.join(LOG_DIR, f"{algo}_seed{seed}_job_log.csv")
    step_counter = 0
    last_save_step = 0
    save_interval = 25000  # Save every 25,000 steps (adjust as needed)
    # -----------------------------------------
    
    # Use the environment's max time for a more accurate progress bar
    progress_bar = tqdm(total=env.episode_max_time, desc=f"Evaluating {algo.upper()}")

    while not (terminated or truncated):
        # Baselines operate on the full view, not just top_k
        gpu_avail = env.gpu_state
        if algo == "fcfs":
            idx = fcfs_choice(env.pending_queue, gpu_avail)
        elif algo == "sjf":
            idx = sjf_choice(env.pending_queue, gpu_avail)
        else: # first_fit
            idx = first_fit_choice(env.pending_queue, gpu_avail)

        # Action is job index if valid, else no-op
        action = idx if idx is not None else env.action_space.n - 1
        obs, r, terminated, truncated, info = env.step(action)

        progress_bar.update(1)
        step_counter += 1
        
        # --- NEW: Check if it's time to save a partial log ---
        if step_counter - last_save_step >= save_interval:
            if env.completed_jobs: # Only save if there's new data
                print(f"\n[Step {step_counter}] Saving partial {algo.upper()} results with {len(env.completed_jobs)} completed jobs...")
                # Note: Assuming _save_eval_outputs saves the completed_jobs log.
                # If you have a more specific function like save_detailed_job_log, use that.
                save_detailed_job_log(env.completed_jobs, log_path) 
                last_save_step = step_counter
        # ----------------------------------------------------

        # Failsafe check to prevent true infinite loops
        if progress_bar.n >= env.episode_max_time and not (terminated or truncated):
            print(f"\n[Failsafe] Evaluation for {algo.upper()} reached max episode time, finishing up.")
            break

    progress_bar.close()
    
    # --- Final save of the complete results ---
    print(f"\n[Final] Saving complete {algo.upper()} evaluation log with {len(env.completed_jobs)} jobs.")
    _save_eval_outputs(env, f"{algo}_seed{seed}")

# ------------------------
# D3QN
# ------------------------

# --- CHANGE: Pass the whole `args` object to access log_suffix ---
def train_d3qn(args, workload):
    seed = args.seed
    resume_path = args.resume
    random.seed(seed)
    np.random.seed(seed)
    env = GPUClusterEnv()

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = D3QNAgent(obs_dim, action_dim, DQN)

    start_ep = 0
    if resume_path and os.path.exists(resume_path):
        agent.load(resume_path)
        print(f"[D3QN] Resumed training from checkpoint: {resume_path}")
        meta = read_checkpoint_meta(resume_path)
        if meta:
            start_ep = int(meta.get("last_completed_episode", -1)) + 1
            print(f"[D3QN] Resuming from episode {start_ep}.")

    last_completed_episode = -1
    try:
        for ep in range(start_ep, TRAIN["max_episodes"]):
            last_completed_episode = ep
            obs, _ = env.reset(seed=seed + ep, workload=workload)
            terminated = truncated = False
            total_reward = 0.0
            num_steps = 0

            # --- ADDITION: Add tqdm progress bar for the inner training loop ---
            max_steps = TRAIN.get("max_steps_per_episode", 5000)
            with tqdm(total=max_steps, desc=f"D3QN Episode {ep}/{TRAIN['max_episodes']}") as pbar:
                while not (terminated or truncated):
                    action = agent.select_action(obs)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done_flag = terminated or truncated
                    agent.store_transition(obs, action, reward, next_obs, done_flag)
                    agent.update()
                    obs = next_obs
                    total_reward += reward
                    num_steps += 1

                    pbar.update(1)
                    pbar.set_postfix(reward=f"{total_reward:.2f}")
                    if num_steps >= max_steps:
                        truncated = True # Manually truncate if max steps are reached

            # Logging (Summary)
            completed_jobs = [j for j in env.completed_jobs if j.finish_time is not None]
            avg_jct = compute_avg_jct(completed_jobs)
            makespan = max([j.finish_time for j in completed_jobs]) if completed_jobs else 0.0
            utilization = compute_utilization(env, makespan) if makespan > 0 else 0.0
            print(f"[D3QN] ep={ep}, reward={total_reward:.2f}, avg_jct={avg_jct:.2f}, jobs={len(completed_jobs)}")
                
            summary = {
                "algo": "d3qn",
                "seed": seed, "episode": ep, "avg_jct": avg_jct,
                "makespan": makespan, "utilization": utilization,
                "total_reward": total_reward, "num_jobs": len(completed_jobs),
            }
            append_summary(summary, os.path.join(LOG_DIR, f"summary_train{args.log_suffix}.csv"))

            # --- CHANGE: Add detailed per-job logging for this episode ---
            save_detailed_job_log(
                completed_jobs=env.completed_jobs,
                episode=ep,
                log_dir=LOG_DIR,
                prefix=f"d3qn_seed{seed}{args.log_suffix}"
            )

            # Periodic saves
            if (ep + 1) % TRAIN["save_interval"] == 0:
                model_path = os.path.join(MODEL_DIR, f"d3qn_seed{seed}_ep{ep+1}.pt")
                save_model_with_meta(agent, model_path, last_completed_episode, seed)

    except KeyboardInterrupt:
        print("[D3QN] KeyboardInterrupt — saving and exiting.")
    finally:
        final_path = os.path.join(MODEL_DIR, f"d3qn_seed{seed}_final.pt")
        save_model_with_meta(agent, final_path, last_completed_episode, seed)


def evaluate_d3qn(pt_path, seed, workload):
    env = GPUClusterEnv()
    obs, _ = env.reset(seed=seed, workload=workload)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = D3QNAgent(obs_dim, action_dim, DQN)
    agent.load(pt_path)

    terminated = truncated = False
    while not (terminated or truncated):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q = agent.policy_net(obs_t)
        action = int(q.argmax(dim=1).item())
        obs, r, terminated, truncated, info = env.step(action)

    _save_eval_outputs(env, "d3qn_eval")

# ------------------------
# SAC
# -----------------------
def train_sac(args):
    seed = args.seed
    resume_path = args.resume
    random.seed(seed)
    np.random.seed(seed)
    env = GPUClusterEnv()

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = SACAgent(obs_dim, action_dim, SAC)

    start_ep = 0
    if resume_path and os.path.exists(resume_path):
        agent.load(resume_path)
        print(f"[SAC] Resumed training from checkpoint: {resume_path}")
        meta = read_checkpoint_meta(resume_path)
        # --- THIS IS THE KEY LOGIC CHANGE ---
        if meta and not args.reset_episode_counter:
            start_ep = int(meta.get("last_completed_episode", -1)) + 1
            print(f"[SAC] Continuing from episode {start_ep}.")
        else:
            print("[SAC] Starting fine-tuning run from episode 0.")
            # start_ep remains 0

    last_completed_episode = -1
    try:
        for ep in range(start_ep, TRAIN["max_episodes"]):
            last_completed_episode = ep
            # Generate a NEW random workload for THIS episode
            workload = synthetic_workload_generator(max_time=TRAIN["max_steps_per_episode"])
            obs, _ = env.reset(seed=seed + ep, workload=workload)
            terminated = truncated = False
            total_reward = 0.0
            num_steps = 0

            # --- ADDITION: Add tqdm progress bar for the inner training loop ---
            max_steps = TRAIN.get("max_steps_per_episode", 5000)
            with tqdm(total=max_steps, desc=f"SAC Episode {ep}/{TRAIN['max_episodes']}") as pbar:
                while not (terminated or truncated):
                    action = agent.select_action(obs, deterministic=False)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done_flag = terminated or truncated
                    agent.store_transition(obs, action, reward, next_obs, done_flag)
                    agent.update()
                    obs = next_obs
                    total_reward += reward
                    num_steps += 1

                    pbar.update(1)
                    pbar.set_postfix(reward=f"{total_reward:.2f}")
                    if num_steps >= max_steps:
                        truncated = True # Manually truncate if max steps are reached
            
            # Logging (Summary)
            completed_jobs = [j for j in env.completed_jobs if j.finish_time is not None]
            avg_jct = compute_avg_jct(completed_jobs)
            makespan = max([j.finish_time for j in completed_jobs]) if completed_jobs else 0.0
            utilization = compute_utilization(env, makespan) if makespan > 0 else 0.0
            print(f"[SAC] ep={ep}, reward={total_reward:.2f}, avg_jct={avg_jct:.2f}, jobs={len(completed_jobs)}")
            
            summary = {
                "algo": "sac",
                "seed": seed, "episode": ep, "avg_jct": avg_jct,
                "makespan": makespan, "utilization": utilization,
                "total_reward": total_reward, "num_jobs": len(completed_jobs),
            }
            append_summary(summary, os.path.join(LOG_DIR, f"summary_train{args.log_suffix}.csv"))

            # --- CHANGE: Add detailed per-job logging for this episode ---
            save_detailed_job_log(
                completed_jobs=env.completed_jobs,
                episode=ep,
                log_dir=LOG_DIR,
                prefix=f"sac_seed{seed}{args.log_suffix}"
            )

            # Periodic saves
            if (ep + 1) % TRAIN["save_interval"] == 0:
                model_path = os.path.join(MODEL_DIR, f"sac_seed{seed}_ep{ep+1}.pt")
                save_model_with_meta(agent, model_path, last_completed_episode, seed)

    except KeyboardInterrupt:
        print("[SAC] KeyboardInterrupt — saving and exiting.")
    finally:
        final_path = os.path.join(MODEL_DIR, f"sac_seed{seed}_final.pt")
        save_model_with_meta(agent, final_path, last_completed_episode, seed)


def evaluate_sac(pt_path, seed, workload):
    env = GPUClusterEnv()
    obs, _ = env.reset(seed=seed, workload=workload)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = SACAgent(obs_dim, action_dim, SAC)
    agent.load(pt_path)

    terminated = truncated = False
    
    # --- NEW: Parameters for partial saving ---
    log_path = os.path.join(LOG_DIR, f"sac_eval_seed{seed}_job_log.csv")
    step_counter = 0
    last_save_step = 0
    save_interval = 25000  # Save every 25,000 steps (adjust as needed)
    # -----------------------------------------

    # Use the environment's max time for the progress bar
    progress_bar = tqdm(total=env.episode_max_time, desc="Evaluating SAC")

    while not (terminated or truncated):
        action = agent.select_action(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(action)
        
        progress_bar.update(1)
        step_counter += 1

        # --- NEW: Check if it's time to save a partial log ---
        if step_counter - last_save_step >= save_interval:
            if env.completed_jobs:
                print(f"\n[Step {step_counter}] Saving partial SAC results with {len(env.completed_jobs)} completed jobs...")
                # Assuming you have a function like _save_eval_outputs or save_detailed_job_log
                save_detailed_job_log(env.completed_jobs, log_path) 
                last_save_step = step_counter
        # ----------------------------------------------------

        # Failsafe check (optional but good to keep)
        if progress_bar.n >= env.episode_max_time and not (terminated or truncated):
            print("\n[Failsafe] Evaluation reached max episode time, finishing up.")
            break

    progress_bar.close()
    
    # --- Final save of the complete results ---
    print(f"\n[Final] Saving complete SAC evaluation log with {len(env.completed_jobs)} jobs.")
    # The function _save_eval_outputs should also use a specific path
    _save_eval_outputs(env, f"sac_eval_seed{seed}") 



# ------------------------
# Entry point
# ------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["d3qn", "sac", "fcfs", "sjf", "first_fit"], default=TRAIN["algo"])
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--seed", type=int, default=TRAIN["seed"])
    parser.add_argument("--resume", type=str, help="Path to checkpoint file to resume training from.")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file for evaluation.")
    # --- CHANGE: Add log_suffix argument for unique log filenames ---
    parser.add_argument("--log_suffix", type=str, default="", help="Suffix for log files to keep them unique.")
    
    # --- ADD THIS NEW ARGUMENT ---
    parser.add_argument("--reset_episode_counter", action="store_true",
                        help="Reset episode counter to 0 when resuming training. Useful for fine-tuning.")
    # Workload arguments
    parser.add_argument("--workload_type", choices=["synthetic", "google", "alibaba"], default="synthetic",
                        help="Type of workload to use.")
    parser.add_argument("--workload_path", type=str, default=None,
                        help="Path to the workload trace file (required for google/alibaba).")
    parser.add_argument("--arrival_lambda", type=float, default=2.5, help="Lambda for job arrival rate.")
    
    args = parser.parse_args()

    # --- Use the default heterogeneous configuration from config.py ---
    from config import DEFAULT_CLUSTER_CFG
    print(" Running in HETEROGENEOUS cluster mode.")
    cluster_cfg = DEFAULT_CLUSTER_CFG

    # --- Load Workload ---
    workload = None
    if args.workload_type == "synthetic":
        print("Using synthetic workload generator.")
        workload = synthetic_workload_generator(max_time=TRAIN["max_steps_per_episode"])
    else:
        if not args.workload_path:
            raise ValueError(f"Must provide --workload_path for workload_type '{args.workload_type}'")
        print(f"Loading workload from trace: {args.workload_path}")
        workload = parse_workload_trace(args.workload_path, args.workload_type)
    
    if not workload:
        raise RuntimeError("Workload could not be loaded or generated.")
    
    # --- Execute Mode ---
    if args.mode == "train":
        if args.algo == "d3qn":
            # --- CHANGE: Pass the whole `args` object ---
            train_d3qn(args, workload=workload)
        elif args.algo == "sac":
            # --- CHANGE: Pass the whole `args` object ---
            train_sac(args)
        else:
            print("Training is not implemented for baselines.")

    elif args.mode == "eval":
        if args.algo == "d3qn":
            pt = args.checkpoint or os.path.join(MODEL_DIR, f"d3qn_seed{args.seed}_final.pt")
            evaluate_d3qn(pt, seed=args.seed, workload=workload)
        elif args.algo == "sac":
            pt = args.checkpoint or os.path.join(MODEL_DIR, f"sac_seed{args.seed}_final.pt")
            evaluate_sac(pt, seed=args.seed, workload=workload)
        elif args.algo in ["fcfs", "sjf", "first_fit"]:
            run_baseline(args.algo, args.seed, workload=workload)
        else:
            print("Evaluation not implemented for this algorithm.")

if __name__ == "__main__":
    main()
