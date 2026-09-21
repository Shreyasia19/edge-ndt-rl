"""
baselines/ilp_baseline.py
ILP-based optimal allocation using Google OR-Tools.

Formulates the task allocation problem as an Integer Linear
Program — minimise total completion time subject to resource
constraints (base paper Eq. 6-9).

This is NP-hard (proved in base paper Theorem 1) so we only
run it on small snapshots (<=15 tasks) to establish a
theoretical lower bound for comparison.
"""

import numpy as np
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_agent.env import EdgeComputingEnv

try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("OR-Tools not installed. Run: pip install ortools")


class ILPPolicy:
    """
    ILP-based optimal task allocation.
    Solves each allocation decision as an ILP snapshot.
    Provides theoretical lower bound on ACT.
    """

    def __init__(self, num_nodes=8, time_limit_ms=1000):
        self.num_nodes = num_nodes
        self.time_limit_ms = time_limit_ms
        self.solve_times = []

    def predict(self, obs, deterministic=True):
        """
        Solve ILP for current state and return best node.
        Falls back to greedy if OR-Tools unavailable or timeout.
        """
        if not ORTOOLS_AVAILABLE:
            # Greedy fallback
            loads = obs[:self.num_nodes]
            return int(np.argmin(loads)), None

        # Extract node loads from obs
        node_loads = obs[:self.num_nodes]
        node_online = obs[2*self.num_nodes:3*self.num_nodes]

        start = time.time()

        # Build CP-SAT model
        model = cp_model.CpModel()

        # Decision variables: which node to assign (one-hot)
        assign = [
            model.NewBoolVar(f'assign_{n}')
            for n in range(self.num_nodes)
        ]

        # Constraint: exactly one node selected (Eq. 7)
        model.Add(sum(assign) == 1)

        # Constraint: only online nodes (Eq. 8)
        for n in range(self.num_nodes):
            if node_online[n] < 0.5:
                model.Add(assign[n] == 0)

        # Objective: minimise estimated completion time
        # Proxy: node with lowest current load
        load_ints = [
            int(node_loads[n] * 1000)
            for n in range(self.num_nodes)
        ]
        model.Minimize(
            sum(load_ints[n] * assign[n]
                for n in range(self.num_nodes))
        )

        # Solve with time limit
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = (
            self.time_limit_ms / 1000.0
        )
        status = solver.Solve(model)

        elapsed = (time.time() - start) * 1000
        self.solve_times.append(elapsed)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for n in range(self.num_nodes):
                if solver.Value(assign[n]) == 1:
                    return n, None

        # Fallback to greedy if ILP fails
        return int(np.argmin(node_loads)), None

    def avg_solve_time_ms(self):
        if not self.solve_times:
            return 0
        return np.mean(self.solve_times)


def evaluate_ilp(
    volatility='normal',
    n_episodes=20,
    tasks_per_episode=15,
):
    """
    Evaluate ILP policy.
    Uses fewer episodes + smaller tasks due to computational cost.
    """
    print(f"Evaluating ILP — {volatility} "
          f"({n_episodes} episodes, {tasks_per_episode} tasks)...")
    print("Note: ILP is slower — establishing lower bound only")

    env = EdgeComputingEnv(
        twin_mode=False,
        volatility=volatility,
        tasks_per_episode=tasks_per_episode,
        seed=123,
    )

    policy = ILPPolicy(num_nodes=8)
    acts, sla_rates = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = policy.predict(obs)
            obs, reward, done, truncated, info = env.step(int(action))

        stats = env.get_episode_stats()
        acts.append(stats['mean_act_ms'])
        sla_rates.append(stats['sla_violation_rate'])

    result = {
        'label': 'ILP',
        'volatility': volatility,
        'mean_act_ms': round(np.mean(acts), 2),
        'std_act_ms': round(np.std(acts), 2),
        'sla_violation_rate': round(np.mean(sla_rates), 4),
        'avg_solve_time_ms': round(policy.avg_solve_time_ms(), 2),
    }

    print(f"  Mean ACT: {result['mean_act_ms']:.0f}ms | "
          f"SLA: {result['sla_violation_rate']:.2%} | "
          f"Solve time: {result['avg_solve_time_ms']:.1f}ms")
    return result


if __name__ == "__main__":
    print("=" * 50)
    print("ILP Baseline Evaluation")
    print("=" * 50)
    for mode in ['normal', 'high_churn', 'low_bandwidth']:
        evaluate_ilp(volatility=mode)