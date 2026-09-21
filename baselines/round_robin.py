"""
baselines/round_robin.py
Round Robin baseline — cycles through nodes in order.
Simplest possible allocation strategy.
Used as lower-bound baseline in comparison table.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_agent.env import EdgeComputingEnv


class RoundRobinPolicy:
    """Assigns tasks to nodes in round-robin order."""

    def __init__(self, num_nodes=8):
        self.num_nodes = num_nodes
        self.current = 0

    def predict(self, obs, deterministic=True):
        """Match Stable-Baselines3 predict() interface."""
        node = self.current % self.num_nodes
        self.current += 1
        return node, None

    def reset(self):
        self.current = 0


def evaluate_round_robin(
    volatility='normal',
    n_episodes=50,
    tasks_per_episode=20,
):
    """Evaluate round robin on real simulator."""
    print(f"Evaluating Round Robin — {volatility}...")

    env = EdgeComputingEnv(
        twin_mode=False,
        volatility=volatility,
        tasks_per_episode=tasks_per_episode,
        seed=123,
    )

    policy = RoundRobinPolicy(num_nodes=8)
    acts, sla_rates = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        policy.reset()
        done = False

        while not done:
            action, _ = policy.predict(obs)
            obs, reward, done, truncated, info = env.step(action)

        stats = env.get_episode_stats()
        acts.append(stats['mean_act_ms'])
        sla_rates.append(stats['sla_violation_rate'])

    result = {
        'label': 'Round Robin',
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
        evaluate_round_robin(volatility=mode)