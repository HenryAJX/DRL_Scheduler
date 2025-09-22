# metrics/plots.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any

# NOTE: do not set global matplotlib styles/colors here (respect instruction).
# Each function will create and save its own figures.

def _ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def jct_vs_arrival_plot(jobs, outdir, prefix="eval", bins=200):
    """
    Scatter plot: job arrival_time vs job completion time (JCT).
    Color by priority and optionally plot a rolling average.
    """
    _ensure_dir(outdir)
    df = pd.DataFrame([{
        "job_id": j.job_id,
        "user_id": j.user_id,
        "arrival": j.arrival_time,
        "start": j.start_time if j.start_time is not None else np.nan,
        "finish": j.finish_time if j.finish_time is not None else np.nan,
        "runtime": j.runtime,
        "priority": j.priority,
        "flops": j.flops
    } for j in jobs])
    df = df.dropna(subset=["finish", "arrival"])
    if df.empty:
        print("[plots] no finished jobs to plot jct_vs_arrival")
        return
    df["jct"] = df["finish"] - df["arrival"]

    # --- Scatter Plot (No Changes Here) ---
    priorities = sorted(df["priority"].unique())
    priority_map = {p: i for i, p in enumerate(priorities)}
    df["pidx"] = df["priority"].map(priority_map)
    plt.figure(figsize=(8, 5))
    sc = plt.scatter(df["arrival"], df["jct"], c=df["pidx"], alpha=0.6, s=10)
    plt.xlabel("Arrival time (s)")
    plt.ylabel("Job Completion Time (JCT) (s)")
    plt.title("JCT vs Arrival Time")
    handles = []
    for p in priorities:
        handles.append(plt.Line2D([], [], marker="o", linestyle="None",
                                  label=f"priority={p}"))
    plt.legend(handles=handles, loc="upper right", fontsize="small")
    path = os.path.join(outdir, f"{prefix}_jct_vs_arrival.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"[plots] Saved JCT vs arrival -> {path}")

    # --- Binned Mean Plot (Corrected Logic) ---
    try:
        if df.empty or len(df) < bins:
             print("[plots] Not enough data to produce binned mean JCT.")
             return

        df_sorted = df.sort_values("arrival")
        
        # Use pd.cut to create the bins. This returns a Series of Intervals.
        arrival_bins = pd.cut(df_sorted["arrival"], bins=bins)
        
        # Group by these bins and calculate the mean JCT for each.
        bmeans = df_sorted.groupby(arrival_bins)["jct"].mean()

        # --- THE FIX ---
        # Get the midpoints of the bins from the index's categories.
        # bmeans.index is a CategoricalIndex.
        # bmeans.index.categories is an IntervalIndex, which has the .mid attribute.
        bin_centers = bmeans.index.categories.mid

        # Plot using the actual time values (bin_centers) for the x-axis.
        plt.figure(figsize=(8, 4))
        plt.plot(bin_centers, bmeans.values)
        plt.xlabel("Arrival Time Bins (s)")
        plt.ylabel("Mean JCT (s)")
        plt.title("Binned Mean JCT over Arrival Time")
        bpath = os.path.join(outdir, f"{prefix}_jct_arrival_binned.png")
        plt.tight_layout()
        plt.savefig(bpath)
        plt.close()
        print(f"[plots] Saved binned mean JCT -> {bpath}")
    except Exception as e:
        print(f"[plots] failed to produce binned mean JCT: {e}")


def per_priority_jct_plots(jobs, outdir, prefix="eval"):
    """
    Boxplot of JCTs per priority level and a small CSV summarizing stats.
    """
    _ensure_dir(outdir)
    rows = []
    for j in jobs:
        if j.finish_time is None:
            continue
        rows.append({"priority": j.priority, "jct": j.finish_time - j.arrival_time})
    if not rows:
        print("[plots] no finished jobs for per-priority plot")
        return
    df = pd.DataFrame(rows)
    groups = df.groupby("priority")["jct"]

    # summary CSV
    summary = groups.agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
    csv_path = os.path.join(outdir, f"{prefix}_per_priority_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"[plots] Saved priority summary CSV -> {csv_path}")

    # boxplot
    plt.figure(figsize=(8, 5))
    priorities = summary["priority"].tolist()
    data = [df[df["priority"] == p]["jct"].values for p in priorities]
    plt.boxplot(data, labels=[str(p) for p in priorities], showfliers=False)
    plt.xlabel("Priority")
    plt.ylabel("JCT (s)")
    plt.title("Per-priority JCT distribution (boxplot)")
    out_path = os.path.join(outdir, f"{prefix}_per_priority_boxplot.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[plots] Saved per-priority boxplot -> {out_path}")

def flops_weighted_utilization_plot(jobs, env, outdir, prefix="eval", bins=200):
    """
    Compute FLOPS-weighted utilization over time.

    For each job, instantaneous compute rate (FLOPS/sec) = job.flops / job.runtime
    Busy_flops(t) = sum over jobs active at t of (job.flops/job.runtime)
    Cluster_total_flops = sum(type_count*type_flops) if env has gpu_type_info OR fallback to Env.num_gpus * mean flops

    Returns timeseries plot saved to PNG.
    """
    _ensure_dir(outdir)
    # compute total cluster flops if possible
    total_cluster_flops = None
    try:
        if hasattr(env, "gpu_type_info"):
            total_cluster_flops = sum([info["count"] * info["flops"] for info in env.gpu_type_info.values()])
        elif hasattr(env, "gpu_types"):
            total_cluster_flops = sum([t["count"] * t["flops"] for t in env.gpu_types])
    except Exception:
        total_cluster_flops = None

    # fallback: estimate from average flops in jobs and num_gpus attr
    if total_cluster_flops is None:
        if hasattr(env, "num_gpus"):
            avg_flops = np.mean([getattr(j, "flops", 0.0) for j in jobs]) if jobs else 1.0
            total_cluster_flops = getattr(env, "num_gpus", 1) * avg_flops
        else:
            total_cluster_flops = max(1.0, np.mean([getattr(j, "flops", 1.0) for j in jobs]))

    # determine makespan
    finishes = [j.finish_time for j in jobs if j.finish_time is not None]
    if not finishes:
        print("[plots] no finished jobs for flops utilization")
        return
    makespan = max(finishes)
    times = np.linspace(0.0, makespan, bins + 1)
    busy_flops_ts = np.zeros_like(times)
    # for each job add its flops rate to bins where it's active
    for j in jobs:
        if j.start_time is None or j.finish_time is None:
            continue
        # instantaneous FLOPS/sec for job:
        rate = float(j.flops) / max(1.0, j.runtime)
        # find bin indices where job is active
        active_mask = (times >= j.start_time) & (times <= j.finish_time)
        busy_flops_ts[active_mask] += rate

    util_ts = busy_flops_ts / (total_cluster_flops + 1e-12)
    plt.figure(figsize=(8, 4))
    plt.plot(times, util_ts)
    plt.xlabel("Time (s)")
    plt.ylabel("FLOPS-weighted Utilization (fraction)")
    plt.title("FLOPS-weighted Utilization over time")
    out_path = os.path.join(outdir, f"{prefix}_flops_weighted_utilization.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[plots] Saved FLOPS-weighted utilization -> {out_path}")

def combined_jct_cdfs(list_of_job_csvs: List[str], outdir, prefix="agg"):
    """
    Makes combined empirical CDFs of JCTs for multiple runs (list of CSV paths).
    Each CSV should have columns arrival_time/start_time/finish_time etc (like the saved job logs).
    """
    _ensure_dir(outdir)
    plt.figure(figsize=(8, 5))
    legends = []
    all_stats = []
    for csv in list_of_job_csvs:
        try:
            df = pd.read_csv(csv)
            if "finish_time" not in df.columns or "arrival_time" not in df.columns:
                print(f"[plots] CSV {csv} missing required columns")
                continue
            df = df.dropna(subset=["finish_time", "arrival_time"])
            if df.empty:
                continue
            df["jct"] = df["finish_time"] - df["arrival_time"]
            jcts = np.sort(df["jct"].values)
            cdf = np.arange(1, len(jcts) + 1) / float(len(jcts))
            plt.plot(jcts, cdf)
            legends.append(os.path.basename(csv))
            all_stats.append({"csv": csv, "mean_jct": float(df["jct"].mean()), "median_jct": float(df["jct"].median()), "n": len(df)})
        except Exception as e:
            print(f"[plots] failed reading {csv}: {e}")
    if legends:
        plt.xlabel("JCT (s)")
        plt.ylabel("Empirical CDF")
        plt.title("Combined JCT CDFs across runs")
        plt.legend(legends, fontsize="small")
        out_path = os.path.join(outdir, f"{prefix}_combined_jct_cdfs.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        stats_csv = os.path.join(outdir, f"{prefix}_combined_stats.csv")
        pd.DataFrame(all_stats).to_csv(stats_csv, index=False)
        print(f"[plots] Saved combined CDFs -> {out_path} and stats -> {stats_csv}")
    else:
        print("[plots] no valid CSVs for combined CDFs")

def aggregate_runs_bar(avg_stats: List[Dict[str, Any]], outdir, prefix="agg"):
    """
    avg_stats: list of dicts containing {'label': str, 'avg_jct': float, 'std_jct': float, 'n': int}
    Produces a bar chart with error bars.
    """
    _ensure_dir(outdir)
    if not avg_stats:
        print("[plots] no runs to aggregate")
        return
    labels = [s["label"] for s in avg_stats]
    means = [s["avg_jct"] for s in avg_stats]
    stds = [s.get("std_jct", 0.0) for s in avg_stats]
    x = np.arange(len(labels))
    plt.figure(figsize=(max(6, len(labels)*1.2), 5))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Average JCT (s)")
    plt.title("Average JCT across runs")
    out_path = os.path.join(outdir, f"{prefix}_avg_jct_bar.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[plots] Saved aggregated avg JCT bar -> {out_path}")
