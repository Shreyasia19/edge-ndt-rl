"""
rl_agent/env.py
Custom Gymnasium environment wrapping the Digital Twin.

The RL agent trains entirely inside this environment —
which uses the twin as its dynamics model (fast, cheap).

MDP Formulation (extends base paper Section IV-A to IV-D):

State:  Graph of edge network — node loads, queues,
        online status, link bandwidths (52-dim vector)

Action: [target_node (discrete), bandwidth_fraction (continuous)]
        — which node to offload task to + how much BW to allocate

Reward: r = -ACT - lambda*SLA_violation - mu*energy_cost
        Extends base paper Eq.14: r_t = -TT_t
        Our extension adds SLA penalty + energy term

Transition: Twin MLP predicts next state (base paper Eq.15)

Episode: One batch of tasks from the trace dataset
         Done when all tasks in batch are processed
"""

import gymnasium as gym
import numpy as np
import torch
import pandas as pd
import os
import sys
from gymnasium import spaces

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from digital_twin.model import MLPDigitalTwin
from network_emulation.topology import (
    EdgeNetworkSimulator, MicroserviceTask
)


class EdgeComputingEnv(gym.Env):
    """
    Custom Gymnasium environment for edge resource orchestration.

    Wraps the digital twin as the transition model during training.
    Can switch to real simulator for sim-to-real evaluation.

    Args:
        twin_mode: if True, use digital twin for transitions (training)
                   if False, use real simulator (evaluation/deployment)
        volatility: network condition ('normal','high_churn','low_bandwidth')
        tasks_per_episode: number of tasks per episode
        num_nodes: number of edge nodes
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        twin_mode: bool = True,
        volatility: str = 'normal',
        tasks_per_episode: int = 20,
        num_nodes: int = 8,
        lambda_sla: float = 0.3,
        mu_energy: float = 0.1,
        model_path: str = 'digital_twin/saved_models/twin_best.pt',
        config_path: str = 'data/processed/config_8nodes.json',
        tasks_path: str = 'data/processed/tasks_train.csv',
        seed: int = 42,
    ):
        super().__init__()

        self.twin_mode = twin_mode
        self.volatility = volatility
        self.tasks_per_episode = tasks_per_episode
        self.num_nodes = num_nodes
        self.lambda_sla = lambda_sla
        self.mu_energy = mu_energy
        self.seed_val = seed

        # State dimension: 3*N node features + L link features
        # N=8 nodes: loads(8) + queues(8) + online(8) = 24
        # L=28 links: bandwidth(28)
        # Total = 52
        self.state_dim = 3 * num_nodes + 28

        # ── Action Space ──────────────────────────────────────
        # Discrete: which node to assign task to (0 to N-1)
        # We use Discrete for simplicity — PPO handles this well
        self.action_space = spaces.Discrete(num_nodes)

        # ── Observation Space ─────────────────────────────────
        # Flat vector of network state + current task features
        # State (52) + task features (4) = 56 dim
        self.task_feat_dim = 4
        obs_dim = self.state_dim + self.task_feat_dim
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # ── Load Digital Twin ─────────────────────────────────
        if twin_mode:
            self.twin = MLPDigitalTwin(
                state_dim=self.state_dim,
                action_dim=2,
                hidden_dim=128,
            )
            if os.path.exists(model_path):
                self.twin.load_state_dict(
                    torch.load(model_path, map_location='cpu')
                )
                self.twin.eval()
            else:
                print(f"Warning: twin model not found at {model_path}")
                print("Using untrained twin — train first!")

        # ── Load Real Simulator ───────────────────────────────
        self.real_sim = EdgeNetworkSimulator(
            config_path=config_path,
            volatility=volatility,
            seed=seed,
        )

        # ── Load Task Dataset ─────────────────────────────────
        self.tasks_df = pd.read_csv(tasks_path)
        self.task_pool = self.tasks_df.to_dict('records')

        # Episode state
        self.current_state = None
        self.current_task = None
        self.task_idx = 0
        self.step_count = 0
        self.episode_tasks = []
        self.episode_rewards = []
        self.episode_completion_times = []
        self.episode_sla_violations = 0

        print(f"EdgeComputingEnv initialized")
        print(f"  Mode: {'Digital Twin' if twin_mode else 'Real Simulator'}")
        print(f"  Volatility: {volatility}")
        print(f"  Obs dim: {obs_dim} | Action space: {num_nodes} nodes")

    # ─────────────────────────────────────────
    # Core Gymnasium methods
    # ─────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """Reset environment for new episode."""
        super().reset(seed=seed)

        # Reset simulator
        self.real_sim.reset()

        # Reset episode tracking
        self.step_count = 0
        self.episode_rewards = []
        self.episode_completion_times = []
        self.episode_sla_violations = 0

        # Sample tasks for this episode
        start_idx = (self.task_idx * self.tasks_per_episode) % \
            len(self.task_pool)
        self.episode_tasks = [
            self.task_pool[(start_idx + i) % len(self.task_pool)]
            for i in range(self.tasks_per_episode)
        ]
        self.task_idx += 1

        # Get initial network state
        self.current_state = self.real_sim.get_state_vector()

        # Load first task
        self.current_task = self._load_task(0)

        obs = self._make_observation(
            self.current_state, self.current_task
        )
        info = {'episode_start': True}

        return obs, info

    def step(self, action: int):
        """
        Execute action (assign task to node), get next state + reward.

        In twin_mode=True: uses digital twin for next state
        In twin_mode=False: uses real simulator
        """
        self.step_count += 1

        # Validate action
        action = int(action)
        action = min(action, self.num_nodes - 1)

        # Default bandwidth fraction (0.5 = half available BW)
        bw_fraction = 0.5

        # ── Execute in real simulator (always for actual metrics) ──
        ct, sla_violated = self.real_sim.assign_task(
            self.current_task, action, bw_fraction
        )
        self.real_sim.step(dt=1.0)

        # ── Get next state ─────────────────────────────────────
        if self.twin_mode:
            # Use twin to predict next state (fast, for training)
            state_tensor = torch.tensor(
                self.current_state, dtype=torch.float32
            )
            action_tensor = torch.tensor(
                [action / self.num_nodes, bw_fraction],
                dtype=torch.float32
            )
            with torch.no_grad():
                next_state_pred, reward_pred = self.twin(
                    state_tensor, action_tensor
                )
            next_state = next_state_pred.numpy()
            # Clip to valid range
            next_state = np.clip(next_state, 0.0, 1.0)
        else:
            # Use real simulator state (for evaluation)
            next_state = self.real_sim.get_state_vector()

        # ── Compute reward ─────────────────────────────────────
        # Extends base paper Eq.14: r_t = -TT_t
        node_load = self.real_sim.nodes[action].current_load

        # Normalise completion time (base paper mean ~350k ms)
        act_norm = min(ct / 500000.0, 5.0)
        sla_penalty = self.lambda_sla if sla_violated else 0.0
        energy_penalty = self.mu_energy * node_load

        reward = -(act_norm + sla_penalty + energy_penalty)

        # Track episode stats
        self.episode_rewards.append(reward)
        self.episode_completion_times.append(ct)
        if sla_violated:
            self.episode_sla_violations += 1

        # ── Check if episode done ──────────────────────────────
        done = self.step_count >= self.tasks_per_episode
        truncated = False

        # ── Load next task ─────────────────────────────────────
        if not done:
            self.current_task = self._load_task(self.step_count)

        # Update state
        self.current_state = next_state

        # Build observation
        obs = self._make_observation(next_state, self.current_task)

        # Info dict (useful for logging)
        info = {
            'completion_time_ms': ct,
            'sla_violated': int(sla_violated),
            'node_load': node_load,
            'assigned_node': action,
        }

        if done:
            info['episode_act'] = np.mean(self.episode_completion_times)
            info['episode_sla_rate'] = (
                self.episode_sla_violations / self.tasks_per_episode
            )
            info['episode_reward'] = sum(self.episode_rewards)

        return obs, reward, done, truncated, info

    # ─────────────────────────────────────────
    # Helper methods
    # ─────────────────────────────────────────

    def _make_observation(
        self,
        state_vec: np.ndarray,
        task: MicroserviceTask
    ) -> np.ndarray:
        """
        Combine network state + current task features into observation.
        Task features: cpu_load, data_size, num_microservices, deadline
        All normalised to [0,1].
        """
        task_features = np.array([
            min(task.cpu_load_kcycles / 1000.0, 1.0),
            min(task.data_size_mbit / 1000.0, 1.0),
            min(task.num_microservices / 50.0, 1.0),
            min(task.deadline_ms / 500000.0, 1.0),
        ], dtype=np.float32)

        obs = np.concatenate([state_vec, task_features])
        return obs.astype(np.float32)

    def _load_task(self, idx: int) -> MicroserviceTask:
        """Load task from episode task list."""
        row = self.episode_tasks[idx % len(self.episode_tasks)]
        return MicroserviceTask(
            task_id=int(row['task_id']),
            service_id=int(row['service_id']),
            cpu_load_kcycles=float(row['cpu_load_kcycles']),
            data_size_mbit=float(row['data_size_mbit']),
            num_microservices=int(row['num_microservices']),
            release_time=float(row['release_time']),
            deadline_ms=float(row['deadline_ms']),
            source_node=int(row['source_node']),
        )

    def get_episode_stats(self) -> dict:
        """Return current episode statistics."""
        if not self.episode_completion_times:
            return {}
        return {
            'mean_act_ms': np.mean(self.episode_completion_times),
            'sla_violation_rate': (
                self.episode_sla_violations / max(self.step_count, 1)
            ),
            'mean_reward': np.mean(self.episode_rewards),
            'total_steps': self.step_count,
        }

    def render(self):
        """Print current network status."""
        stats = self.get_episode_stats()
        print(
            f"Step {self.step_count}/{self.tasks_per_episode} | "
            f"ACT: {stats.get('mean_act_ms', 0):.0f}ms | "
            f"SLA violations: {self.episode_sla_violations}"
        )


# ─────────────────────────────────────────
# Environment test
# ─────────────────────────────────────────

def test_environment():
    """Quick sanity check — run before training."""
    print("=" * 50)
    print("Environment Sanity Check")
    print("=" * 50)

    env = EdgeComputingEnv(
        twin_mode=True,
        volatility='normal',
        tasks_per_episode=10,
    )

    # Test reset
    obs, info = env.reset()
    print(f"\nObservation shape: {obs.shape}")
    print(f"Obs range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"Action space: {env.action_space}")

    # Test random actions
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done:
            break

    stats = env.get_episode_stats()
    print(f"\nRandom policy results (10 steps):")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Mean ACT: {stats['mean_act_ms']:.0f} ms")
    print(f"  SLA violations: {env.episode_sla_violations}/10")
    print(f"  SLA rate: {stats['sla_violation_rate']:.2f}")

    # Verify gymnasium API compliance
    from gymnasium.utils.env_checker import check_env
    print("\nRunning Gymnasium API check...")
    try:
        check_env(env, warn=True)
        print("Gymnasium API check passed ✅")
    except Exception as e:
        print(f"API check warning: {e}")

    print("\nEnvironment test complete!")
    return env


if __name__ == "__main__":
    test_environment()