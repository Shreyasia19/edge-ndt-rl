"""
rl_agent/train_scalability.py
Trains and evaluates PPO + baselines across different node counts.
Run: python3 -m rl_agent.train_scalability

Tests: 4, 6, 8, 10, 12, 16 nodes
For each node count:
  - Retrains digital twin on that topology
  - Trains PPO agent
  - Evaluates against Round Robin + Greedy
  - Records ACT, SLA violation rate
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network_emulation.topology import EdgeNetworkSimulator, MicroserviceTask
from digital_twin.model import MLPDigitalTwin
from rl_agent.env import EdgeComputingEnv

plt.rcParams.update({
    'figure.facecolor': '#0a0e1a', 'axes.facecolor': '#0d1b2a',
    'axes.edgecolor': '#1e3a5f', 'axes.labelcolor': '#90a4ae',
    'xtick.color': '#78909c', 'ytick.color': '#78909c',
    'text.color': '#cfd8dc', 'grid.color': '#1a2744', 'grid.alpha': .5,
    'legend.facecolor': '#0d1b2a', 'legend.edgecolor': '#1e3a5f',
})

NODE_COUNTS   = [4, 6, 8, 10, 12, 16]
VOLATILITY    = "normal"
TRAIN_STEPS   = 80000       # enough for convergence per topology
EVAL_EPISODES = 30
TASKS_PER_EP  = 20
RESULTS_DIR   = "rl_agent/results/scalability"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# QUICK TWIN RETRAIN FOR A GIVEN NODE COUNT
# ══════════════════════════════════════════════════════════════════════

def collect_telemetry_for_nodes(n_nodes: int,
                                 n_episodes: int = 60,
                                 steps_per_ep: int = 50):
    """Collect (s, a, s') tuples for this node count."""
    import torch
    print(f"  Collecting telemetry for {n_nodes} nodes…")

    config_path = f"data/processed/config_{n_nodes}nodes.json"
    sim = EdgeNetworkSimulator(
        config_path=config_path,
        volatility=VOLATILITY,
        seed=42
    )
    tasks_df = pd.read_csv("data/processed/tasks_train.csv")
    task_pool = tasks_df.to_dict("records")
    state_dim = sim.state_dim()
    records = []
    task_idx = 0

    for ep in range(n_episodes):
        sim.reset()
        for step in range(steps_per_ep):
            sv = sim.get_state_vector()
            row = task_pool[task_idx % len(task_pool)]
            task_idx += 1
            task = MicroserviceTask(
                task_id=int(row["task_id"]),
                service_id=int(row["service_id"]),
                cpu_load_kcycles=float(row["cpu_load_kcycles"]),
                data_size_mbit=float(row["data_size_mbit"]),
                num_microservices=int(row["num_microservices"]),
                release_time=float(row["release_time"]),
                deadline_ms=float(row["deadline_ms"]),
                source_node=int(row["source_node"]) % n_nodes,
            )
            online = sim.online_nodes()
            target = np.random.choice(online)
            bw = np.random.uniform(0.1, 0.9)
            ct, viol = sim.assign_task(task, target, bw)
            node_load = sim.nodes[target].current_load
            reward = -min(ct / 500000, 5) - (0.3 if viol else 0) - 0.1 * node_load
            sim.step(dt=1.0)
            nsv = sim.get_state_vector()
            rec = {
                "action_node": target,
                "action_bw":   round(bw, 4),
                "reward":      round(reward, 4),
            }
            for i, v in enumerate(sv):
                rec[f"s_{i}"] = round(float(v), 4)
            for i, v in enumerate(nsv):
                rec[f"ns_{i}"] = round(float(v), 4)
            records.append(rec)

    return pd.DataFrame(records), state_dim


def train_twin_for_nodes(n_nodes: int) -> MLPDigitalTwin:
    """Train a fresh twin for this node count."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    df, state_dim = collect_telemetry_for_nodes(n_nodes)
    print(f"  Training twin — state_dim={state_dim}, records={len(df)}")

    sc = [f"s_{i}"  for i in range(state_dim)]
    nc = [f"ns_{i}" for i in range(state_dim)]
    S  = torch.tensor(df[sc].values,  dtype=torch.float32)
    NS = torch.tensor(df[nc].values,  dtype=torch.float32)
    A  = torch.tensor(df[["action_node","action_bw"]].values,
                      dtype=torch.float32)
    A[:, 0] /= n_nodes
    R  = torch.tensor(df["reward"].values, dtype=torch.float32)

    split = int(len(S) * 0.8)
    dl = DataLoader(TensorDataset(S[:split], A[:split], NS[:split]),
                    batch_size=64, shuffle=True)

    model  = MLPDigitalTwin(state_dim=state_dim, action_dim=2, hidden_dim=128)
    opt    = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(80):
        model.train()
        for s_b, a_b, ns_b in dl:
            opt.zero_grad()
            pred, rp = model(s_b, a_b)
            loss = loss_fn(pred, ns_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    # Save
    save_path = f"{RESULTS_DIR}/twin_{n_nodes}nodes.pt"
    import torch
    torch.save(model.state_dict(), save_path)
    print(f"  Twin saved → {save_path}")
    return model, state_dim


# ══════════════════════════════════════════════════════════════════════
# ENVIRONMENT FACTORY FOR N NODES
# ══════════════════════════════════════════════════════════════════════

def make_env_n(n_nodes: int, twin_path: str,
               state_dim: int, rank: int = 0):
    def _init():
        env = EdgeComputingEnv(
            twin_mode=True,
            volatility=VOLATILITY,
            tasks_per_episode=TASKS_PER_EP,
            num_nodes=n_nodes,
            model_path=twin_path,
            config_path=f"data/processed/config_{n_nodes}nodes.json",
            tasks_path="data/processed/tasks_train.csv",
            seed=42 + rank,
        )
        return Monitor(env)
    return _init


# ══════════════════════════════════════════════════════════════════════
# EVALUATE A POLICY OR BASELINE
# ══════════════════════════════════════════════════════════════════════

def evaluate(policy_or_str, n_nodes: int,
             n_episodes: int = EVAL_EPISODES) -> dict:
    """
    policy_or_str: PPO model OR 'round_robin' OR 'greedy'
    """
    env = EdgeComputingEnv(
        twin_mode=False,
        volatility=VOLATILITY,
        tasks_per_episode=TASKS_PER_EP,
        num_nodes=n_nodes,
        config_path=f"data/processed/config_{n_nodes}nodes.json",
        tasks_path="data/processed/tasks_train.csv",
        seed=999,
    )

    acts, slas = [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        while not done:
            if isinstance(policy_or_str, str):
                if policy_or_str == "round_robin":
                    action = step % n_nodes
                else:  # greedy
                    loads = obs[:n_nodes]
                    action = int(np.argmin(loads))
            else:
                action, _ = policy_or_str.predict(obs, deterministic=True)
                action = int(action)
            obs, _, done, _, _ = env.step(action)
            step += 1
        stats = env.get_episode_stats()
        acts.append(stats["mean_act_ms"])
        slas.append(stats["sla_violation_rate"])

    return {
        "mean_act_ms": round(np.mean(acts), 2),
        "std_act_ms":  round(np.std(acts), 2),
        "sla_rate":    round(np.mean(slas), 4),
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════

def main():
    all_results = []

    for n in NODE_COUNTS:
        print(f"\n{'='*55}")
        print(f"Node count: {n}")
        print(f"{'='*55}")

        # 1. Train twin
        twin_model, state_dim = train_twin_for_nodes(n)
        twin_path = f"{RESULTS_DIR}/twin_{n}nodes.pt"

        # 2. Train PPO
        print(f"  Training PPO — {TRAIN_STEPS:,} steps…")
        train_env = make_vec_env(
            make_env_n(n, twin_path, state_dim), n_envs=2
        )
        ppo = PPO(
            "MlpPolicy", train_env,
            learning_rate=0.0003, n_steps=512,
            batch_size=64, n_epochs=10,
            gamma=0.9, gae_lambda=0.95,
            ent_coef=0.01, verbose=0,
        )
        ppo.learn(total_timesteps=TRAIN_STEPS, progress_bar=True)
        ppo_path = f"{RESULTS_DIR}/ppo_{n}nodes"
        ppo.save(ppo_path)
        print(f"  PPO saved → {ppo_path}.zip")

        # 3. Evaluate all methods
        print(f"  Evaluating…")
        ppo_r  = evaluate(ppo,           n)
        rr_r   = evaluate("round_robin", n)
        gr_r   = evaluate("greedy",      n)

        print(f"  PPO:         ACT={ppo_r['mean_act_ms']:.0f}ms  SLA={ppo_r['sla_rate']:.2%}")
        print(f"  Round Robin: ACT={rr_r['mean_act_ms']:.0f}ms  SLA={rr_r['sla_rate']:.2%}")
        print(f"  Greedy:      ACT={gr_r['mean_act_ms']:.0f}ms  SLA={gr_r['sla_rate']:.2%}")

        for method, res in [("PPO",ppo_r),("Round Robin",rr_r),("Greedy",gr_r)]:
            all_results.append({
                "n_nodes": n,
                "method":  method,
                **res,
            })

    # Save CSV
    df = pd.DataFrame(all_results)
    csv_path = f"{RESULTS_DIR}/scalability_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")
    print(df.to_string(index=False))

    # Plot
    _plot(df, RESULTS_DIR)


def _plot(df: pd.DataFrame, save_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {"PPO": "#4fc3f7", "Round Robin": "#ef5350", "Greedy": "#ffa726"}

    for method, color in colors.items():
        sub = df[df["method"] == method].sort_values("n_nodes")
        axes[0].plot(sub["n_nodes"], sub["mean_act_ms"] / 1000,
                     marker="o", color=color, linewidth=2.5,
                     markersize=7, label=method)
        axes[0].fill_between(
            sub["n_nodes"],
            (sub["mean_act_ms"] - sub["std_act_ms"]) / 1000,
            (sub["mean_act_ms"] + sub["std_act_ms"]) / 1000,
            alpha=0.15, color=color
        )
        axes[1].plot(sub["n_nodes"], sub["sla_rate"] * 100,
                     marker="s", color=color, linewidth=2.5,
                     markersize=7, label=method)

    for ax, title, ylabel in zip(
        axes,
        ["Average Completion Time vs Node Count",
         "SLA Violation Rate vs Node Count"],
        ["Mean ACT (seconds)", "SLA Violation Rate (%)"]
    ):
        ax.set_xlabel("Number of Edge Nodes", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontweight="bold", color="#4fc3f7", fontsize=11)
        ax.set_xticks(NODE_COUNTS)
        ax.legend(fontsize=9)
        ax.grid(True)

    # Add annotation at 8 nodes
    ppo_8 = df[(df["method"] == "PPO") & (df["n_nodes"] == 8)]
    if not ppo_8.empty:
        v = ppo_8["mean_act_ms"].values[0] / 1000
        axes[0].annotate(
            "8 nodes\n(baseline)",
            xy=(8, v), xytext=(9, v + 20),
            fontsize=8, color="#4fc3f7",
            arrowprops=dict(arrowstyle="->", color="#4fc3f7", lw=1.2)
        )

    plt.suptitle(
        f"Scalability Analysis — {VOLATILITY} network condition\n"
        f"PPO vs Round Robin vs Greedy across {len(NODE_COUNTS)} topologies",
        color="#4fc3f7", fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plot_path = f"{save_dir}/scalability_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {plot_path}")


if __name__ == "__main__":
    main()