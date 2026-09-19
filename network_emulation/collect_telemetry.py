"""
collect_telemetry.py
Runs the edge network simulator and collects (state, action,
next_state, reward) tuples for training the digital twin.

This is the data generation pipeline that bridges the real
emulated network and the digital twin model.

Ref: Chen et al., MASS 2023 — Algorithm 1 (Digital Twin
Modelling Algorithm) requires transaction dataset (bs, ba, bs_)
and reward dataset (bs_, r). We generate exactly that here.
"""

import numpy as np
import pandas as pd
import json
import os
import time
from tqdm import tqdm

from topology import EdgeNetworkSimulator, MicroserviceTask

np.random.seed(42)

PROCESSED_DIR = "../data/processed"
TELEMETRY_DIR = "../data/processed/telemetry"
os.makedirs(TELEMETRY_DIR, exist_ok=True)


# ─────────────────────────────────────────
# Random policy (for data collection)
# ─────────────────────────────────────────

def random_policy(sim: EdgeNetworkSimulator):
    """
    Random action selection for data collection.
    We use random actions (not a trained policy) to collect
    diverse (s, a, s') tuples — standard practice for
    model-based RL data collection.
    """
    online = sim.online_nodes()
    if not online:
        return 0, 0.5
    target_node = np.random.choice(online)
    bandwidth_fraction = np.random.uniform(0.1, 0.9)
    return target_node, bandwidth_fraction


# ─────────────────────────────────────────
# Reward function
# ─────────────────────────────────────────

def compute_reward(
    completion_time_ms: float,
    sla_violated: bool,
    node_load: float,
    lambda_sla: float = 0.3,
    mu_energy: float = 0.1
) -> float:
    """
    Reward function for the RL agent.
    Extends base paper Eq. 14: r_t = -TT_t
    We add SLA penalty and energy term as our extension.

    r = -ACT - lambda * SLA_violation - mu * energy_proxy
    """
    # Normalise completion time to [-1, 0] range
    act_penalty = -min(completion_time_ms / 100000.0, 5.0)

    # SLA violation penalty
    sla_penalty = -lambda_sla if sla_violated else 0.0

    # Energy proxy: high node load = high energy cost
    energy_penalty = -mu_energy * node_load

    return act_penalty + sla_penalty + energy_penalty


# ─────────────────────────────────────────
# Collection loop
# ─────────────────────────────────────────

def collect_telemetry(
    config_path: str,
    volatility: str,
    num_episodes: int = 100,
    steps_per_episode: int = 50,
    tasks_per_step: int = 3,
) -> pd.DataFrame:
    """
    Collect (state, action, next_state, reward) tuples
    by running the simulator with a random policy.

    Args:
        config_path: path to network JSON config
        volatility: 'normal' | 'high_churn' | 'low_bandwidth'
        num_episodes: number of full episodes to run
        steps_per_episode: time steps per episode
        tasks_per_step: tasks arriving per time step

    Returns:
        DataFrame with one row per (s, a, s', r) tuple
    """
    sim = EdgeNetworkSimulator(
        config_path=config_path,
        volatility=volatility
    )

    # Load task data
    tasks_df = pd.read_csv(
        os.path.join(PROCESSED_DIR, "tasks_train.csv")
    )
    task_pool = tasks_df.to_dict('records')
    task_idx = 0

    records = []
    state_dim = sim.state_dim()

    print(f"\nCollecting telemetry — volatility: {volatility}")
    print(f"Episodes: {num_episodes}, Steps: {steps_per_episode}")
    print(f"State dim: {state_dim}")

    for episode in tqdm(range(num_episodes), desc=f"{volatility}"):
        sim.reset()

        for step in range(steps_per_episode):
            # Get current state
            state_vec = sim.get_state_vector()

            # Process tasks arriving this step
            for _ in range(tasks_per_step):
                if task_idx >= len(task_pool):
                    task_idx = 0  # cycle through tasks

                row = task_pool[task_idx % len(task_pool)]
                task_idx += 1

                task = MicroserviceTask(
                    task_id=int(row['task_id']),
                    service_id=int(row['service_id']),
                    cpu_load_kcycles=float(row['cpu_load_kcycles']),
                    data_size_mbit=float(row['data_size_mbit']),
                    num_microservices=int(row['num_microservices']),
                    release_time=float(row['release_time']),
                    deadline_ms=float(row['deadline_ms']),
                    source_node=int(row['source_node']),
                )

                # Take random action
                target_node, bw_fraction = random_policy(sim)

                # Execute action in simulator
                completion_time, sla_violated = sim.assign_task(
                    task, target_node, bw_fraction
                )

                # Compute reward
                node_load = sim.nodes[target_node].current_load
                reward = compute_reward(
                    completion_time,
                    sla_violated,
                    node_load
                )

                # Advance time
                sim.step(dt=1.0)

                # Get next state
                next_state_vec = sim.get_state_vector()

                # Log the tuple
                record = {
                    'episode': episode,
                    'step': step,
                    'action_node': target_node,
                    'action_bw': round(bw_fraction, 4),
                    'completion_time_ms': round(completion_time, 2),
                    'sla_violated': int(sla_violated),
                    'reward': round(reward, 4),
                    'volatility': volatility,
                }

                # Add state features
                for i, v in enumerate(state_vec):
                    record[f's_{i}'] = round(float(v), 4)

                # Add next state features
                for i, v in enumerate(next_state_vec):
                    record[f'ns_{i}'] = round(float(v), 4)

                records.append(record)

    return pd.DataFrame(records)


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("=" * 55)
    print("Telemetry Collection Pipeline")
    print("=" * 55)

    start_time = time.time()
    all_dfs = []

    configs = {
        'normal': 'config_8nodes.json',
        'high_churn': 'config_8nodes.json',
        'low_bandwidth': 'config_8nodes.json',
    }

    for volatility, config_file in configs.items():
        config_path = os.path.join(PROCESSED_DIR, config_file)
        df = collect_telemetry(
            config_path=config_path,
            volatility=volatility,
            num_episodes=300,
            steps_per_episode=50,
            tasks_per_step=3,
        )
        all_dfs.append(df)

        # Save per-volatility telemetry
        out_path = os.path.join(
            TELEMETRY_DIR, f"telemetry_{volatility}.csv"
        )
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} records → {out_path}")

    # Save combined telemetry
    combined = pd.concat(all_dfs, ignore_index=True)
    combined_path = os.path.join(
        TELEMETRY_DIR, "telemetry_all.csv"
    )
    combined.to_csv(combined_path, index=False)

    elapsed = time.time() - start_time

    print("\n" + "=" * 55)
    print("Collection Complete")
    print("=" * 55)
    print(f"Total records: {len(combined):,}")
    print(f"State dimension: 52 features per state")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"\nBreakdown:")
    print(combined.groupby('volatility')[
        ['completion_time_ms', 'sla_violated', 'reward']
    ].mean().round(2))
    print(f"\nFiles saved to: {TELEMETRY_DIR}")
    print("\nTelemetry ready for digital twin training!")