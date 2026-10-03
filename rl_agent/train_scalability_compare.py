"""
rl_agent/train_scalability_compare.py

One script that answers: "why 8 nodes?"
  - Fewer nodes (4, 6): is the network too constrained?
  - More nodes (10, 12, 16, 20, 50): does it scale, and do returns diminish?
  - Compares PPO against one other ML model (A2C) plus Round Robin / Greedy.

Design choices that avoid the earlier errors:
  * Self-contained: trains its own twin per node count and saves a PLAIN
    state_dict (no wrapped checkpoint), so env.py loads it cleanly.
  * Nested topologies: node i and link (i,j) have the same parameters in every
    network size, so the 8-node network is a subset of the 20-node network.
    Differences between sizes are then caused by node count, not random configs.
  * Writes configs to data/processed/scal/ (does not touch your existing ones).
  * Saves the CSV after every node count and resumes if you re-run it.

Run:   python3 -m rl_agent.train_scalability_compare
Quick: NODES=4,8,20 TRAIN_STEPS=20000 python3 -m rl_agent.train_scalability_compare
"""

import os
import sys
import json
import time
import inspect
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network_emulation.topology import EdgeNetworkSimulator, MicroserviceTask
from digital_twin.model import MLPDigitalTwin
from rl_agent.env import EdgeComputingEnv

# ── Settings ──────────────────────────────────────────────────────────
NODE_COUNTS = [int(x) for x in
               os.environ.get("NODES", "4,6,8,10,12,16,20,50").split(",")]
TRAIN_STEPS = int(os.environ.get("TRAIN_STEPS", "60000"))
EVAL_EPISODES = int(os.environ.get("EVAL_EPISODES", "30"))
VOLATILITY = "normal"
TASKS_PER_EP = 20
CONFIG_DIR = "data/processed/scal"
RESULTS_DIR = "rl_agent/results/scalability_compare"
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

ALGOS = {
    "PPO": (PPO, dict(learning_rate=3e-4, n_steps=512, batch_size=64,
                      n_epochs=10, gamma=0.9, gae_lambda=0.95,
                      ent_coef=0.01)),
    "A2C": (A2C, dict(learning_rate=7e-4, n_steps=32, gamma=0.9,
                      gae_lambda=0.95, ent_coef=0.01)),
}
COLORS = {"PPO": "#1f77b4", "A2C": "#2ca02c",
          "Round Robin": "#d62728", "Greedy": "#ff7f0e"}


# ══════════════════════════════════════════════════════════════════════
# Preflight: env.py must not hardcode 28 links
# ══════════════════════════════════════════════════════════════════════

def preflight():
    src = inspect.getsource(EdgeComputingEnv.__init__)
    if "+ 28" in src:
        raise SystemExit(
            "\nenv.py still hardcodes 28 links (8-node only). Fix with:\n"
            "  sed -i 's/3 \\* num_nodes + 28/3 * num_nodes + "
            "num_nodes * (num_nodes - 1) \\/\\/ 2/' rl_agent/env.py\n"
            "then re-run.\n"
        )


# ══════════════════════════════════════════════════════════════════════
# Nested network configs
# ══════════════════════════════════════════════════════════════════════

def gen_config(n_nodes: int) -> str:
    nodes = []
    for i in range(n_nodes):
        r = np.random.default_rng(1000 + i)
        nodes.append({
            "node_id": i,
            "compute_capacity_mcps": round(max(5, r.normal(40, 32)), 2),
            "memory_gb": round(max(0.5, r.normal(4, 2)), 2),
        })
    links = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            r = np.random.default_rng(100000 + i * 1000 + j)
            links.append({
                "source": i, "target": j,
                "bandwidth_mbps": round(max(0.5, r.normal(10, 8)), 2),
                "latency_ms": round(max(0.1, r.normal(5, 2.5)), 2),
            })
    path = f"{CONFIG_DIR}/config_{n_nodes}nodes.json"
    with open(path, "w") as f:
        json.dump({"num_nodes": n_nodes, "nodes": nodes, "links": links},
                  f, indent=2)
    return path


# ══════════════════════════════════════════════════════════════════════
# Twin per node count
# ══════════════════════════════════════════════════════════════════════

def collect_telemetry(n_nodes, config_path, n_episodes=60, steps_per_ep=50):
    sim = EdgeNetworkSimulator(config_path=config_path,
                               volatility=VOLATILITY, seed=42)
    task_pool = pd.read_csv("data/processed/tasks_train.csv").to_dict("records")
    state_dim = sim.state_dim()
    rng = np.random.default_rng(0)
    records, task_idx = [], 0

    for _ in range(n_episodes):
        sim.reset()
        for _ in range(steps_per_ep):
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
            target = int(rng.choice(online))
            bw = float(rng.uniform(0.1, 0.9))
            ct, viol = sim.assign_task(task, target, bw)
            load = sim.nodes[target].current_load
            reward = -min(ct / 500000, 5) - (0.3 if viol else 0) - 0.1 * load
            sim.step(dt=1.0)
            nsv = sim.get_state_vector()
            rec = {"action_node": target, "action_bw": bw, "reward": reward}
            rec.update({f"s_{i}": float(v) for i, v in enumerate(sv)})
            rec.update({f"ns_{i}": float(v) for i, v in enumerate(nsv)})
            records.append(rec)
    return pd.DataFrame(records), state_dim


def train_twin(n_nodes, config_path):
    df, state_dim = collect_telemetry(n_nodes, config_path)
    print(f"  Twin: state_dim={state_dim}, records={len(df)}")

    S = torch.tensor(df[[f"s_{i}" for i in range(state_dim)]].values,
                     dtype=torch.float32)
    NS = torch.tensor(df[[f"ns_{i}" for i in range(state_dim)]].values,
                      dtype=torch.float32)
    A = torch.tensor(df[["action_node", "action_bw"]].values,
                     dtype=torch.float32)
    A[:, 0] /= n_nodes

    split = int(len(S) * 0.8)
    dl = DataLoader(TensorDataset(S[:split], A[:split], NS[:split]),
                    batch_size=64, shuffle=True)
    model = MLPDigitalTwin(state_dim=state_dim, action_dim=2, hidden_dim=128)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for _ in range(80):
        model.train()
        for s_b, a_b, ns_b in dl:
            opt.zero_grad()
            pred, _ = model(s_b, a_b)
            loss_fn(pred, ns_b).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    with torch.no_grad():
        pred, _ = model(S[split:], A[split:])
        val_mse = loss_fn(pred, NS[split:]).item()

    path = f"{RESULTS_DIR}/twin_{n_nodes}nodes.pt"
    torch.save(model.state_dict(), path)      # plain state_dict
    print(f"  Twin saved -> {path} (val MSE {val_mse:.5f})")
    return path, state_dim, val_mse


# ══════════════════════════════════════════════════════════════════════
# Env factory + evaluation
# ══════════════════════════════════════════════════════════════════════

def make_env_fn(n_nodes, twin_path, config_path, rank=0):
    def _init():
        env = EdgeComputingEnv(
            twin_mode=True, volatility=VOLATILITY,
            tasks_per_episode=TASKS_PER_EP, num_nodes=n_nodes,
            model_path=twin_path, config_path=config_path,
            tasks_path="data/processed/tasks_train.csv",
            seed=42 + rank,
        )
        return Monitor(env)
    return _init


def evaluate(policy, n_nodes, config_path):
    env = EdgeComputingEnv(
        twin_mode=False, volatility=VOLATILITY,
        tasks_per_episode=TASKS_PER_EP, num_nodes=n_nodes,
        config_path=config_path,
        tasks_path="data/processed/tasks_train.csv", seed=999,
    )
    acts, slas = [], []
    for _ in range(EVAL_EPISODES):
        obs, _ = env.reset()
        done, step = False, 0
        while not done:
            if policy == "round_robin":
                action = step % n_nodes
            elif policy == "greedy":
                action = int(np.argmin(obs[:n_nodes]))
            else:
                action, _ = policy.predict(obs, deterministic=True)
                action = int(action)
            obs, _, done, _, _ = env.step(action)
            step += 1
        st = env.get_episode_stats()
        acts.append(st["mean_act_ms"])
        slas.append(st["sla_violation_rate"])
    return {"mean_act_ms": float(np.mean(acts)),
            "std_act_ms": float(np.std(acts)),
            "sla_rate": float(np.mean(slas))}


# ══════════════════════════════════════════════════════════════════════
# Analysis + plots
# ══════════════════════════════════════════════════════════════════════

def analyse(df):
    print("\n" + "=" * 70)
    print("MEAN ACT (seconds) BY NODE COUNT")
    print("=" * 70)
    piv = (df.pivot(index="n_nodes", columns="method", values="mean_act_ms")
           / 1000).round(1)
    print(piv.to_string())

    if "PPO" in piv.columns and "Round Robin" in piv.columns:
        imp = ((piv["Round Robin"] - piv["PPO"]) / piv["Round Robin"] * 100)
        print("\nPPO improvement over Round Robin (%):")
        print(imp.round(1).to_string())

    best = piv[[c for c in ("PPO", "A2C") if c in piv.columns]].min(axis=1)
    print("\nMarginal effect of adding nodes (best learned policy):")
    prev_n, prev_v = None, None
    for n, v in best.items():
        if prev_n is not None:
            d = (v - prev_v) / prev_v * 100
            per_node = d / (n - prev_n)
            print(f"  {prev_n:>2} -> {n:>2} nodes: {d:+6.1f}% ACT "
                  f"({per_node:+.2f}% per added node)")
        prev_n, prev_v = n, v
    print("\nRead this as: if the % change per added node flattens out "
          "around 8, that is your 'knee' / justification. If it does not, "
          "say so honestly and justify 8 by the base paper's setup.")


def plot(df, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for method, color in COLORS.items():
        sub = df[df["method"] == method].sort_values("n_nodes")
        if sub.empty:
            continue
        axes[0].errorbar(sub["n_nodes"], sub["mean_act_ms"] / 1000,
                         yerr=sub["std_act_ms"] / 1000, marker="o",
                         color=color, label=method, capsize=3, linewidth=2)
        axes[1].plot(sub["n_nodes"], sub["sla_rate"] * 100, marker="s",
                     color=color, label=method, linewidth=2)

    rr = df[df["method"] == "Round Robin"].set_index("n_nodes")["mean_act_ms"]
    for m in ("PPO", "A2C"):
        sub = df[df["method"] == m].set_index("n_nodes")["mean_act_ms"]
        common = sub.index.intersection(rr.index)
        if len(common):
            imp = (rr[common] - sub[common]) / rr[common] * 100
            axes[2].plot(common, imp.values, marker="o",
                         color=COLORS[m], label=f"{m} vs Round Robin",
                         linewidth=2)

    titles = ["Mean completion time vs node count",
              "SLA violation rate vs node count",
              "Improvement over Round Robin"]
    ylabels = ["Mean ACT (s)", "SLA violation (%)", "ACT reduction (%)"]
    for ax, t, yl in zip(axes, titles, ylabels):
        ax.set_xscale("log")
        ax.set_xticks(sorted(df["n_nodes"].unique()))
        ax.set_xticklabels(sorted(df["n_nodes"].unique()))
        ax.axvline(8, color="gray", linestyle="--", alpha=0.6)
        ax.set_xlabel("Number of edge nodes (log scale)")
        ax.set_ylabel(yl)
        ax.set_title(t, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.suptitle("Scalability and algorithm comparison "
                 f"({VOLATILITY} network; dashed line = 8-node baseline)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved -> {path}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    preflight()
    csv_path = f"{RESULTS_DIR}/scalability_compare.csv"
    rows = []
    done_nodes = set()
    if os.path.exists(csv_path):
        old = pd.read_csv(csv_path)
        rows = old.to_dict("records")
        done_nodes = set(old["n_nodes"].unique())
        print(f"Resuming. Already done: {sorted(done_nodes)}")

    t_all = time.time()
    for n in NODE_COUNTS:
        if n in done_nodes:
            continue
        print("\n" + "=" * 60)
        print(f"Node count: {n}")
        print("=" * 60)
        cfg_path = gen_config(n)

        twin_path, state_dim, val_mse = train_twin(n, cfg_path)

        for name, (cls, kwargs) in ALGOS.items():
            print(f"  Training {name} - {TRAIN_STEPS:,} steps...")
            t0 = time.time()
            venv = make_vec_env(make_env_fn(n, twin_path, cfg_path),
                                n_envs=2)
            model = cls("MlpPolicy", venv, verbose=0, **kwargs)
            model.learn(total_timesteps=TRAIN_STEPS, progress_bar=True)
            train_s = time.time() - t0
            model.save(f"{RESULTS_DIR}/{name.lower()}_{n}nodes")
            res = evaluate(model, n, cfg_path)
            print(f"  {name}: ACT={res['mean_act_ms']:.0f} ms  "
                  f"SLA={res['sla_rate']:.2%}  (train {train_s/60:.1f} min)")
            rows.append({"n_nodes": n, "method": name, **res,
                         "train_seconds": round(train_s, 1),
                         "twin_val_mse": val_mse})

        for name, key in (("Round Robin", "round_robin"),
                          ("Greedy", "greedy")):
            res = evaluate(key, n, cfg_path)
            print(f"  {name}: ACT={res['mean_act_ms']:.0f} ms  "
                  f"SLA={res['sla_rate']:.2%}")
            rows.append({"n_nodes": n, "method": name, **res,
                         "train_seconds": 0.0, "twin_val_mse": val_mse})

        pd.DataFrame(rows).to_csv(csv_path, index=False)   # save per n
        print(f"  (saved progress -> {csv_path})")

    df = pd.DataFrame(rows).sort_values(["n_nodes", "method"])
    df.to_csv(csv_path, index=False)
    analyse(df)
    plot(df, f"{RESULTS_DIR}/scalability_compare.png")
    print(f"\nTotal time: {(time.time() - t_all) / 60:.1f} min")
    print(f"Results: {csv_path}")


if __name__ == "__main__":
    main()
