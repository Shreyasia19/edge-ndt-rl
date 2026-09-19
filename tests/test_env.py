"""Unit tests for EdgeComputingEnv."""
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_project_structure():
    """Check all required files exist."""
    required = [
        'network_emulation/topology.py',
        'network_emulation/trace_replay.py',
        'network_emulation/collect_telemetry.py',
        'digital_twin/model.py',
        'digital_twin/train_twin.py',
        'digital_twin/evaluate_twin.py',
        'rl_agent/env.py',
        'baselines/round_robin.py',
    ]
    for path in required:
        assert os.path.exists(path), f"Missing: {path}"


def test_env_imports():
    """Check environment can be imported."""
    from rl_agent.env import EdgeComputingEnv
    assert EdgeComputingEnv is not None


def test_env_spaces():
    """Check observation and action spaces are correct."""
    from rl_agent.env import EdgeComputingEnv
    env = EdgeComputingEnv(
        twin_mode=False,
        tasks_per_episode=5,
    )
    assert env.observation_space.shape == (56,)
    assert env.action_space.n == 8


def test_env_reset():
    """Check reset returns correct observation shape."""
    from rl_agent.env import EdgeComputingEnv
    env = EdgeComputingEnv(
        twin_mode=False,
        tasks_per_episode=5,
    )
    obs, info = env.reset()
    assert obs.shape == (56,)
    assert obs.min() >= 0.0
    assert obs.max() <= 1.0


def test_env_step():
    """Check step returns correct types."""
    from rl_agent.env import EdgeComputingEnv
    env = EdgeComputingEnv(
        twin_mode=False,
        tasks_per_episode=5,
    )
    obs, _ = env.reset()
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    assert obs.shape == (56,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert 'completion_time_ms' in info


def test_env_full_episode():
    """Run a complete episode with random actions."""
    from rl_agent.env import EdgeComputingEnv
    env = EdgeComputingEnv(
        twin_mode=False,
        tasks_per_episode=10,
    )
    obs, _ = env.reset()
    total_reward = 0
    done = False
    steps = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
    assert steps == 10
    assert total_reward < 0  # reward is always negative (penalty)
    stats = env.get_episode_stats()
    assert 'mean_act_ms' in stats
    assert stats['mean_act_ms'] > 0
