"""
rl_agent/train_scalability_rf.py

Scalability + algorithm comparison with a classical ML baseline.

Methods compared at every node count:
  Round Robin | Greedy (min load) | Fastest Node (static) |
  Random Forest (supervised ML) | A2C | PPO

Random Forest scheduler ("predict-then-pick"):
  - Trained on (state, task, candidate node) -> log(completion time)
    collected from the simulator under random assignments.
  - At decision time it predicts completion time for every online node and
    picks the lowest. Same data source as the twin; no RL involved.

Multiple seeds are used so "PPO is better" can be tested, not just asserted.

Run:   python3 -m rl_agent.train_scalability_rf
Quick: NODES=4,8,20 TRAIN_STEPS=20000 SEEDS=2 python3 -m rl_agent.train_scalability_rf
Needs: pip3 install scikit-learn scipy
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

from sklearn.ensemble import RandomForestRegressor
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

try:
    from scipy import stats as sps
except ImportError:
    sps = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network_emulation.topology import EdgeNetworkSimulator, MicroserviceTask
from digital_twin.model import MLPDigitalTwin
from rl_agent.env import EdgeComputingEnv

# ── Settings ──────────────────────────────────────────────────────────
NODE_COUNTS = [int(x) for x in
               os.environ.get("NODES", "4,6,8,10,12,16,20,50").split(",")]
TRAIN_STEPS = int(os.environ.get("TRAIN_STEPS", "60000"))
EVAL_EPISODES = int(os.environ.get("EVAL_EPISODES", "30"))
N_SEEDS = int(os.environ.get("SEEDS", "3"))
VOLATILITY = "normal"
TASKS_PER_EP = 20
BW_FRACTION = 0.5          # env.py uses a fixed 0.5 bandwidth fraction
CONFIG_DIR = "data/processed/scal"
RESULTS_DIR = "rl_agent/results/scalability_rf"
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

ALGOS = {
    "PPO": (PPO, dict(learning_rate=3e-4, n_steps=512, batch_size=64,
                      n_epochs=10, gamma=0.9, gae_lambda=0.95,
                      ent_coef=0.01)),
    "A2C": (A2C, dict(learning_rate=7e-4, n_steps=32, gamma=0.9,
                      gae_lambda=0.95, ent_coef=0.01)),
}
ORDER = ["Round Robin", "Greedy", "Fastest Node", "Random Forest",
         "A2C", "PPO"]
COLORS = {"Round Robin": "#d62728", "Greedy": "#ff7f0e",
          "Fastest Node": "#8c564b", "Random Forest": "#9467bd",
          "A2C": "#2ca02c", "PPO": "#1f77b4"}


# ══════════════════════════════════════════════════════════════════════
# Preflight
# ══════════════════════════════════════════════════════════════════════

def preflight():
    src = inspect.getsource(EdgeComputingEnv.__init__)
    if "+ 28" in src:
        raise SystemExit(
            "\nenv.py still hardcodes 28 links (8-node only). Fix with:\n"
            "  sed -i 's/3 \\* num_nodes + 28/3 * num_nodes + "
            "num_nodes * (num_nodes - 1) \\/\\/ 2/' rl_agent/env.py\n"
        )


# ══════════════════════════════════════════════════════════════════════
# Nested network configs (node i / link i-j identical across sizes)
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


def load_static(cfg_path, n):
    """Per-node capacity and mean link bandwidth from the config."""
    cfg = json.load(open(cfg_path))
    caps = np.array([x["compute_capacity_mcps"] for x in cfg["nodes"]],
                    dtype=float)
    bw = np.zeros(n)
    for l in cfg["links"]:
        bw[l["source"]] += l["bandwidth_mbps"]
        bw[l["target"]] += l["bandwidth_mbps"]
    return caps, bw / max(n - 1, 1)


# ══════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════

def make_task(row, n_nodes):
    return MicroserviceTask(
        task_id=int(row["task_id"]), service_id=int(row["service_id"]),
        cpu_load_kcycles=float(row["cpu_load_kcycles"]),
        data_size_mbit=float(row["data_size_mbit"]),
        num_microservices=int(row["num_microservices"]),
        release_time=float(row["release_time"]),
        deadline_ms=float(row["deadline_ms"]),
        source_node=int(row["source_node"]) % n_nodes,
    )


def task_features(task):
    """Same normalisation as EdgeComputingEnv._make_observation."""
    return np.array([
        min(task.cpu_load_kcycles / 1000.0, 1.0),
        min(task.data_size_mbit / 1000.0, 1.0),
        min(task.num_microservices / 50.0, 1.0),
        min(task.deadline_ms / 500000.0, 1.0),
    ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════
# Digital twin (one per node count, used for RL training)
# ══════════════════════════════════════════════════════════════════════

def pick_node(rng, sim):
    online = list(sim.online_nodes())
    if not online:
        online = list(range(len(sim.nodes)))
    return int(rng.choice(online))


def collect_telemetry(n_nodes, cfg_path, n_episodes=60, steps_per_ep=50):
    sim = EdgeNetworkSimulator(config_path=cfg_path,
                               volatility=VOLATILITY, seed=42)
    pool = pd.read_csv("data/processed/tasks_train.csv").to_dict("records")
    state_dim = sim.state_dim()
    rng = np.random.default_rng(0)
    records, idx = [], 0
    for _ in range(n_episodes):
        sim.reset()
        for _ in range(steps_per_ep):
            sv = sim.get_state_vector()
            task = make_task(pool[idx % len(pool)], n_nodes)
            idx += 1
            target = pick_node(rng, sim)
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


def train_twin(n_nodes, cfg_path):
    df, state_dim = collect_telemetry(n_nodes, cfg_path)
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
    torch.save(model.state_dict(), path)          # plain state_dict
    print(f"  Twin: state_dim={state_dim} val MSE {val_mse:.5f} -> {path}")
    return path, val_mse


# ══════════════════════════════════════════════════════════════════════
# Random Forest scheduler
# ══════════════════════════════════════════════════════════════════════

def rf_features(obs, n, caps, mean_bw):
    """One feature row per candidate node, from an env-style observation."""
    loads, queues = obs[:n], obs[n:2 * n]
    online = obs[2 * n:3 * n]
    cpu, data, nms, dl = obs[-4:]
    ones = np.ones(n)
    return np.column_stack([
        loads, queues, online, caps, mean_bw,
        ones * cpu, ones * data, ones * nms, ones * dl,
        cpu * 1000.0 / caps,                         # compute demand / capacity
        data * 1000.0 / np.maximum(mean_bw, 0.1),    # data / bandwidth
        ones * loads.mean(),
    ])


def collect_rf_data(n, cfg_path, caps, mean_bw, seed,
                    n_episodes=250):
    """(features of chosen node) -> log1p(completion time), random policy."""
    sim = EdgeNetworkSimulator(config_path=cfg_path,
                               volatility=VOLATILITY, seed=100 + seed)
    pool = pd.read_csv("data/processed/tasks_train.csv").to_dict("records")
    rng = np.random.default_rng(seed)
    X, y = [], []
    idx = int(rng.integers(0, len(pool)))
    for _ in range(n_episodes):
        sim.reset()
        for _ in range(TASKS_PER_EP):
            sv = sim.get_state_vector()
            task = make_task(pool[idx % len(pool)], n)
            idx += 1
            obs = np.concatenate([sv, task_features(task)])
            online = list(sim.online_nodes())
            if not online:
                sim.step(dt=1.0)
                continue
            target = int(rng.choice(online))
            feats = rf_features(obs, n, caps, mean_bw)[target]
            ct, _ = sim.assign_task(task, target, BW_FRACTION)
            sim.step(dt=1.0)
            X.append(feats)
            y.append(np.log1p(ct))
    return np.array(X), np.array(y)


class RFPolicy:
    def __init__(self, rf, n, caps, mean_bw):
        self.rf, self.n, self.caps, self.mean_bw = rf, n, caps, mean_bw

    def predict_node(self, obs):
        pred = self.rf.predict(rf_features(obs, self.n, self.caps,
                                           self.mean_bw))
        pred = np.where(obs[2 * self.n:3 * self.n] > 0.5, pred, np.inf)
        return int(np.argmin(pred))


def train_rf(n, cfg_path, caps, mean_bw, seed):
    X, y = collect_rf_data(n, cfg_path, caps, mean_bw, seed)
    split = int(len(X) * 0.8)
    rf = RandomForestRegressor(n_estimators=150, max_depth=14,
                               min_samples_leaf=3, n_jobs=-1,
                               random_state=seed)
    rf.fit(X[:split], y[:split])
    r2 = rf.score(X[split:], y[split:])
    return RFPolicy(rf, n, caps, mean_bw), r2


# ══════════════════════════════════════════════════════════════════════
# Env factory + evaluation
# ══════════════════════════════════════════════════════════════════════

def make_env_fn(n_nodes, twin_path, cfg_path, base_seed, rank=0):
    def _init():
        env = EdgeComputingEnv(
            twin_mode=True, volatility=VOLATILITY,
            tasks_per_episode=TASKS_PER_EP, num_nodes=n_nodes,
            model_path=twin_path, config_path=cfg_path,
            tasks_path="data/processed/tasks_train.csv",
            seed=base_seed + rank,
        )
        return Monitor(env)
    return _init


def evaluate(policy, n, cfg_path, caps):
    env = EdgeComputingEnv(
        twin_mode=False, volatility=VOLATILITY,
        tasks_per_episode=TASKS_PER_EP, num_nodes=n,
        config_path=cfg_path,
        tasks_path="data/processed/tasks_train.csv", seed=999,
    )
    acts, slas = [], []
    counts = np.zeros(n)
    for _ in range(EVAL_EPISODES):
        obs, _ = env.reset()
        done, step = False, 0
        while not done:
            if policy == "round_robin":
                a = step % n
            elif policy == "greedy":
                a = int(np.argmin(obs[:n]))
            elif policy == "fastest":
                a = int(np.argmax(np.where(obs[2 * n:3 * n] > 0.5,
                                           caps, -1.0)))
            elif hasattr(policy, "predict_node"):
                a = policy.predict_node(obs)
            else:
                a, _ = policy.predict(obs, deterministic=True)
                a = int(a)
            counts[a] += 1
            obs, _, done, _, _ = env.step(a)
            step += 1
        st = env.get_episode_stats()
        acts.append(st["mean_act_ms"])
        slas.append(st["sla_violation_rate"])
    return {"mean_act_ms": float(np.mean(acts)),
            "std_act_ms": float(np.std(acts)),
            "sla_rate": float(np.mean(slas)),
            "top_node_share": float(counts.max() / counts.sum())}


# ══════════════════════════════════════════════════════════════════════
# Analysis + plots
# ══════════════════════════════════════════════════════════════════════

def analyse(df):
    g = df.groupby(["n_nodes", "method"])["mean_act_ms"]
    mean = g.mean().unstack() / 1000
    std = g.std().unstack().fillna(0) / 1000
    cols = [c for c in ORDER if c in mean.columns]

    print("\n" + "=" * 78)
    print("MEAN ACT (seconds) +/- std across seeds")
    print("=" * 78)
    out = pd.DataFrame({c: mean[c].round(1).astype(str) + " +/- "
                        + std[c].round(1).astype(str) for c in cols})
    print(out.to_string())

    print("\nBest method per node count (lowest mean ACT):")
    for n, row in mean[cols].iterrows():
        print(f"  {n:>2} nodes: {row.idxmin()} ({row.min():.1f} s)")

    if "Round Robin" in mean.columns and "PPO" in mean.columns:
        print("\nPPO reduction vs each method (%):")
        for c in cols:
            if c == "PPO":
                continue
            red = (mean[c] - mean["PPO"]) / mean[c] * 100
            print(f"  vs {c:<14}: " +
                  "  ".join(f"{n}:{v:+.0f}%" for n, v in red.items()))

    if "top_node_share" in df.columns:
        print("\nShare of tasks sent to the single most-used node "
              "(near 100% = collapsed policy):")
        sh = df.groupby(["n_nodes", "method"])["top_node_share"] \
            .mean().unstack()[cols].round(2)
        print(sh.to_string())

    if sps is not None and N_SEEDS >= 2:
        print("\nPPO vs others, Welch t-test on per-seed ACT "
              f"(n={N_SEEDS} seeds; use >=5 seeds for a firm claim):")
        for n in sorted(df["n_nodes"].unique()):
            ppo = df[(df.n_nodes == n) & (df.method == "PPO")]["mean_act_ms"]
            for other in ("A2C", "Random Forest"):
                o = df[(df.n_nodes == n) & (df.method == other)]["mean_act_ms"]
                if len(ppo) > 1 and len(o) > 1:
                    t, p = sps.ttest_ind(ppo, o, equal_var=False)
                    verdict = ("PPO lower" if ppo.mean() < o.mean()
                               else "PPO higher")
                    print(f"  {n:>2} nodes vs {other:<13}: p={p:.3f} "
                          f"({verdict}"
                          f"{', significant' if p < 0.05 else ', not significant'})")


def plot(df, path):
    g = df.groupby(["n_nodes", "method"])["mean_act_ms"]
    mean = g.mean().unstack() / 1000
    std = g.std().unstack().fillna(0) / 1000
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    for m in ORDER:
        if m not in mean.columns:
            continue
        axes[0].errorbar(mean.index, mean[m], yerr=std[m], marker="o",
                         color=COLORS[m], label=m, capsize=3, linewidth=2)
    axes[0].set_xscale("log")
    axes[0].set_xticks(list(mean.index))
    axes[0].set_xticklabels(list(mean.index))
    axes[0].axvline(8, color="gray", linestyle="--", alpha=0.6)
    axes[0].set_xlabel("Number of edge nodes (log scale)")
    axes[0].set_ylabel("Mean ACT (s)")
    axes[0].set_title("ACT vs node count (error bars: std across seeds)",
                      fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    ref = 8 if 8 in mean.index else mean.index[len(mean.index) // 2]
    cols = [m for m in ORDER if m in mean.columns]
    axes[1].bar(cols, [mean.loc[ref, m] for m in cols],
                yerr=[std.loc[ref, m] for m in cols],
                color=[COLORS[m] for m in cols], capsize=4)
    axes[1].set_title(f"Method comparison at {ref} nodes",
                      fontweight="bold")
    axes[1].set_ylabel("Mean ACT (s)")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved -> {path}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    preflight()
    csv_path = f"{RESULTS_DIR}/scalability_rf.csv"
    rows, done_nodes = [], set()
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
        caps, mean_bw = load_static(cfg_path, n)
        twin_path, val_mse = train_twin(n, cfg_path)

        # Heuristic baselines (deterministic - evaluated once)
        for name, key in (("Round Robin", "round_robin"),
                          ("Greedy", "greedy"),
                          ("Fastest Node", "fastest")):
            res = evaluate(key, n, cfg_path, caps)
            print(f"  {name:<14} ACT={res['mean_act_ms']:.0f} ms  "
                  f"SLA={res['sla_rate']:.2%}")
            rows.append({"n_nodes": n, "method": name, "seed": 0, **res,
                         "train_seconds": 0.0})

        for s in range(N_SEEDS):
            print(f"  -- seed {s} --")
            # Random Forest
            t0 = time.time()
            rf_pol, r2 = train_rf(n, cfg_path, caps, mean_bw, s)
            res = evaluate(rf_pol, n, cfg_path, caps)
            print(f"  Random Forest  ACT={res['mean_act_ms']:.0f} ms  "
                  f"SLA={res['sla_rate']:.2%}  (holdout R2={r2:.2f}, "
                  f"top-node {res['top_node_share']:.0%})")
            rows.append({"n_nodes": n, "method": "Random Forest", "seed": s,
                         **res, "train_seconds": round(time.time() - t0, 1)})

            # RL algorithms
            for name, (cls, kwargs) in ALGOS.items():
                t0 = time.time()
                venv = make_vec_env(
                    make_env_fn(n, twin_path, cfg_path, 42 + 1000 * s),
                    n_envs=2)
                model = cls("MlpPolicy", venv, verbose=0, seed=s, **kwargs)
                model.learn(total_timesteps=TRAIN_STEPS,
                            progress_bar=False)
                model.save(f"{RESULTS_DIR}/{name.lower()}_{n}nodes_s{s}")
                res = evaluate(model, n, cfg_path, caps)
                print(f"  {name:<14} ACT={res['mean_act_ms']:.0f} ms  "
                      f"SLA={res['sla_rate']:.2%}  "
                      f"(top-node {res['top_node_share']:.0%}, "
                      f"train {(time.time()-t0)/60:.1f} min)")
                rows.append({"n_nodes": n, "method": name, "seed": s, **res,
                             "train_seconds": round(time.time() - t0, 1)})

        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"  (progress saved -> {csv_path})")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    analyse(df)
    plot(df, f"{RESULTS_DIR}/scalability_rf.png")
    print(f"\nTotal time: {(time.time() - t_all) / 60:.1f} min")
    print(f"Results: {csv_path}")


if __name__ == "__main__":
    main()
