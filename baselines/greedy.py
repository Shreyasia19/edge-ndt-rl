"""
baselines/greedy.py
Greedy baseline — always assigns task to least-loaded node.
Matches base paper's Greedy baseline (Section V).
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_agent.env import EdgeComputingEnv


class GreedyPolicy:
    """
    Assigns task to node with lowest current CPU load.
    Matches base paper Greedy: picks device with minimal
    estimated completion time (Section V-A Baselines).
    """

    def __init__(self, num_nodes=8, state_dim=52):
        self.num_nodes = num_nodes
        # Node loads are first N elements of state vector
        self.load_slice = slice(0, num_nodes)

    def predict(self, obs, deterministic=True):
        """Pick least loaded online node."""
        node_loads = obs[self.load_slice]
        # Pick node with minimum load
        best_node = int(np.argmin(node_loads))
        return best_node, None


def evaluate_greedy(
    volatility='normal',
    n_episodes=50,
    tasks_per_episode=20,
):
    """Evaluate greedy policy on real simulator."""
    print(f"Evaluating Greedy — {volatility}...")

    env = EdgeComputingEnv(
        twin_mode=False,
        volatility=volatility,
        tasks_per_episode=tasks_per_episode,
        seed=123,
    )

    policy = GreedyPolicy(num_nodes=8)
    acts, sla_rates = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = policy.predict(obs)
            obs, reward, done, truncated, info = env.step(action)

        stats = env.get_episode_stats()
        acts.append(stats['mean_act_ms'])
        sla_rates.append(stats['sla_violation_rate'])

    result = {
        'label': 'Greedy',
        'volatility': volatility,
        'mean_act_ms': round(np.mean(acts), 2),
        'std_act_ms': round(np.std(acts), 2),
        'sla_violation_rate': round(np.mean(sla_rates), 4),
    }

    print(f"  Mean ACT: {result['mean_act_ms']:.0f}ms | "
          f"SLA: {result['sla_violation_rate']:.2%}")
    return result


if __name__ == "__main__":
    for mode in ['normal', 'high_churn', 'low_bandwidth']:
        evaluate_greedy(volatility=mode)