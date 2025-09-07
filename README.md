# Deep Reinforcement Learning for GPU Cluster Scheduling

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides a simulation platform to train and evaluate Deep Reinforcement Learning (DRL) agents for scheduling jobs in a heterogeneous GPU cluster. The environment is built using **SimPy** for discrete-event simulation and adheres to the **Gymnasium** (formerly OpenAI Gym) API, making it easy to test and develop different scheduling algorithms.

The primary goal is to develop intelligent schedulers that can minimize job completion times (JCT) and maximize cluster utilization, outperforming traditional heuristic-based methods like First-Come-First-Served (FCFS) and Shortest-Job-First (SJF).

---

## Key Features

-   **DRL Agents:** Includes implementations for Deep Dueling Q-Network (**D3QN**) and Soft Actor-Critic (**SAC**).
-   **Baseline Schedulers:** Provides classic heuristic algorithms like **FCFS** and **SJF** for performance comparison.
-   **Heterogeneous Environment:** The simulation environment can model a cluster with multiple types of GPUs (e.g., A100, V100, T4), each with different performance characteristics (FLOPS).
-   **Flexible Workloads:** Supports both procedurally generated **synthetic workloads** and real-world **trace-based workloads** from Google and Alibaba cluster data.
-   **Comprehensive Metrics:** Logs detailed job-level data and automatically generates summary statistics and plots for:
    -   Job Completion Time (JCT)
    -   Makespan
    -   Cluster Utilization (GPU-hour and FLOPS-weighted)
    -   Per-priority performance analysis

## Project Structure

```
.
├── agents/             # DRL agent implementations (D3QN, SAC)
├── env/                # Gymnasium environment logic (gpu_env.py)
├── workloads/          # Workload generation and trace parsing
├── schedulers/         # Baseline algorithm implementations
├── metrics/            # Logging and plotting utilities
├── models/             # Saved model checkpoints
├── logs/               # Output logs, summary CSVs, and plots
├── config.py           # Central configuration for hyperparameters and environment settings
└── main.py             # Main script for training and evaluation
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/GPU-Cluster-Scheduler-RL.git](https://github.com/your-username/GPU-Cluster-Scheduler-RL.git)
    cd GPU-Cluster-Scheduler-RL
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Linux or macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install the required packages:**
    A `requirements.txt` file is provided.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The `main.py` script is the entry point for all operations. You can specify the algorithm, mode (train/eval), and other parameters via command-line arguments.

### 1. Training a DRL Agent

To train an agent, specify the `algo` and `mode`. Models will be saved periodically to the `/models` directory.

```bash
# Train the SAC agent with a specific random seed
python main.py --mode train --algo sac --seed 42
```

```bash
# Train the D3QN agent and resume from a checkpoint if it exists
python main.py --mode train --algo d3qn --seed 123 --resume models/d3qn_seed123_ep500.pt
```

### 2. Evaluating a Trained Agent

To evaluate a trained model, specify the path to its checkpoint file. The evaluation results, including job logs and plots, will be saved in the `/logs` directory.

```bash
# Evaluate the final SAC model from the training run with seed 42
python main.py --mode eval --algo sac --seed 42 --checkpoint models/sac_seed42_final.pt
```

If `--checkpoint` is omitted, the script will look for a default final model based on the seed.

### 3. Running a Baseline Scheduler

The baseline algorithms can be run directly in `eval` mode.

```bash
# Evaluate the First-Come-First-Served (FCFS) scheduler
python main.py --mode eval --algo fcfs --seed 0
```

```bash
# Evaluate the Shortest-Job-First (SJF) scheduler
python main.py --mode eval --algo sjf --seed 0
```

### 4. Using Different Workloads

By default, a synthetic workload is used. You can also use real-world traces by specifying the workload type and path.

```bash
# Train an agent using a Google cluster trace
python main.py --mode train --algo sac --workload_type google --workload_path /path/to/google_trace.csv
```

## Configuration

All major hyperparameters, environment settings, reward weights, and cluster definitions are centralized in `config.py`. You can modify this file to experiment with:
-   Agent learning rates, batch sizes, etc.
-   The number and type of GPUs in the cluster.
-   The reward function components.
-   Synthetic workload generation parameters.

## Future Work & Contributing

This platform is a foundation for research in DRL-based scheduling. Contributions are welcome! Potential areas for improvement include:
-   Implementing more advanced DRL agents (e.g., PPO, Rainbow).
-   Adding support for job preemption and migration.
-   Expanding the state and action spaces to include more complex cluster information.
-   Developing a web-based dashboard for visualizing results.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
