"""
rl_agent/train_agent.py
PPO agent training on the Digital Twin environment.

Replaces base paper's DDPG with PPO for improved stability.
Targets ~8,000 episodes to match base paper convergence benchmark.

Training flow:
1. Train inside Digital Twin (fast, cheap)
2. Save best policy checkpoint
3. Evaluate on real simulator (sim-to-real transfer)
4. Log everything to W&B for experiment tracking
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor

from rl_agent.env import EdgeComputingEnv

# ─────────────────────────────────────────
# Config — matches base paper where possible
# ─────────────────────────────────────────

CONFIG = {
    # Environment
    'num_nodes': 8,
    'tasks_per_episode': 20,
    'volatility': 'normal',

    # Training
    'algorithm': 'PPO',        # PPO replaces base paper's DDPG
    'total_timesteps': 200000, # ~10,000 episodes × 20 steps
    'n_envs': 4,               # parallel envs for faster training

    # PPO hyperparameters
    'ppo': {
        'learning_rate': 0.0003,
        'n_steps': 512,
        'batch_size': 64,      # same as base paper
        'n_epochs': 10,
        'gamma': 0.9,          # same as base paper
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,      # exploration bonus
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'policy': 'MlpPolicy',
    },

    # Reward weights
    'lambda_sla': 0.3,
    'mu_energy': 0.1,

    # Paths
    'save_dir': 'rl_agent/policies',
    'log_dir': 'rl_agent/logs',
    'results_dir': 'rl_agent/results',
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)
os.makedirs(CONFIG['log_dir'], exist_ok=True)
os.makedirs(CONFIG['results_dir'], exist_ok=True)


# ─────────────────────────────────────────
# Custom callback for detailed logging
# ─────────────────────────────────────────

class TrainingMetricsCallback(BaseCallback):
    """
    Logs ACT, SLA violation rate, and reward per episode.
    These are the metrics we compare against baselines.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_acts = []
        self.episode_sla_rates = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Check for episode end in any parallel env
        for info in self.locals.get('infos', []):
            if 'episode_act' in info:
                self.episode_acts.append(info['episode_act'])
                self.episode_sla_rates.append(
                    info.get('episode_sla_rate', 0)
                )
                self.episode_count += 1

                if self.episode_count % 100 == 0:
                    avg_act = np.mean(self.episode_acts[-100:])
                    avg_sla = np.mean(self.episode_sla_rates[-100:])
                    print(
                        f"Episode {self.episode_count:5d} | "
                        f"Avg ACT: {avg_act:.0f}ms | "
                        f"SLA violation: {avg_sla:.2f}"
                    )
        return True

    def get_metrics(self):
        return {
            'episode_acts': self.episode_acts,
            'episode_sla_rates': self.episode_sla_rates,
        }


# ─────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────

def make_env(volatility='normal', twin_mode=True, rank=0):
    """Create a monitored environment instance."""
    def _init():
        env = EdgeComputingEnv(
            twin_mode=twin_mode,
            volatility=volatility,
            tasks_per_episode=CONFIG['tasks_per_episode'],
            num_nodes=CONFIG['num_nodes'],
            lambda_sla=CONFIG['lambda_sla'],
            mu_energy=CONFIG['mu_energy'],
            seed=42 + rank,
        )
        env = Monitor(env, CONFIG['log_dir'])
        return env
    return _init


# ─────────────────────────────────────────
# Training
# ─────────────────────────────────────────

def train_ppo(config=CONFIG):
    """Train PPO agent on the digital twin environment."""

    print("=" * 60)
    print("PPO Agent Training")
    print("=" * 60)
    print(f"Algorithm: PPO (replaces base paper DDPG)")
    print(f"Total timesteps: {config['total_timesteps']:,}")
    print(f"Parallel envs: {config['n_envs']}")
    print(f"Volatility: {config['volatility']}")
    print(f"Batch size: {config['ppo']['batch_size']} (matches base paper)")
    print(f"Gamma: {config['ppo']['gamma']} (matches base paper)")

    # Create vectorised training environments
    print("\nCreating environments...")
    train_env = make_vec_env(
        make_env(
            volatility=config['volatility'],
            twin_mode=True
        ),
        n_envs=config['n_envs'],
    )

    # Create evaluation environment (real simulator)
    eval_env = Monitor(
        EdgeComputingEnv(
            twin_mode=False,   # real simulator for evaluation
            volatility=config['volatility'],
            tasks_per_episode=config['tasks_per_episode'],
            seed=999,
        )
    )

    # Build PPO model
    print("\nBuilding PPO model...")
    model = PPO(
        policy=config['ppo']['policy'],
        env=train_env,
        learning_rate=config['ppo']['learning_rate'],
        n_steps=config['ppo']['n_steps'],
        batch_size=config['ppo']['batch_size'],
        n_epochs=config['ppo']['n_epochs'],
        gamma=config['ppo']['gamma'],
        gae_lambda=config['ppo']['gae_lambda'],
        clip_range=config['ppo']['clip_range'],
        ent_coef=config['ppo']['ent_coef'],
        vf_coef=config['ppo']['vf_coef'],
        max_grad_norm=config['ppo']['max_grad_norm'],
        verbose=0,
        tensorboard_log=config['log_dir'],
    )

    total_params = sum(
        p.numel() for p in model.policy.parameters()
    )
    print(f"PPO policy parameters: {total_params:,}")

    # Callbacks
    metrics_callback = TrainingMetricsCallback()

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=config['save_dir'],
        name_prefix='ppo_checkpoint',
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config['save_dir'],
        log_path=config['log_dir'],
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    # Train
    print(f"\nTraining started — {config['total_timesteps']:,} timesteps")
    print("-" * 60)

    start_time = datetime.now()

    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=[
            metrics_callback,
            checkpoint_callback,
            eval_callback,
        ],
        progress_bar=True,
    )

    elapsed = (datetime.now() - start_time).seconds
    print(f"\nTraining complete in {elapsed}s")

    # Save final model
    final_path = os.path.join(config['save_dir'], 'ppo_final')
    model.save(final_path)
    print(f"Final model saved → {final_path}.zip")

    # Save config
    config_path = os.path.join(config['save_dir'], 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Plot and save training curves
    metrics = metrics_callback.get_metrics()
    plot_training_curves(metrics, config['results_dir'])

    return model, metrics


# ─────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────

def evaluate_policy(
    model,
    volatility='normal',
    n_episodes=50,
    twin_mode=False,
    label='PPO'
):
    """
    Evaluate trained policy on given environment.
    twin_mode=False → evaluates on real simulator (sim-to-real).
    """
    print(f"\nEvaluating {label} on "
          f"{'twin' if twin_mode else 'real'} network "
          f"({volatility}, {n_episodes} episodes)...")

    env = EdgeComputingEnv(
        twin_mode=twin_mode,
        volatility=volatility,
        tasks_per_episode=CONFIG['tasks_per_episode'],
        seed=123,
    )

    acts, sla_rates, rewards = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            ep_reward += reward

        stats = env.get_episode_stats()
        acts.append(stats['mean_act_ms'])
        sla_rates.append(stats['sla_violation_rate'])
        rewards.append(ep_reward)

    results = {
        'label': label,
        'volatility': volatility,
        'twin_mode': twin_mode,
        'mean_act_ms': round(np.mean(acts), 2),
        'std_act_ms': round(np.std(acts), 2),
        'sla_violation_rate': round(np.mean(sla_rates), 4),
        'mean_reward': round(np.mean(rewards), 4),
        'n_episodes': n_episodes,
    }

    print(f"  Mean ACT:      {results['mean_act_ms']:.0f} ± "
          f"{results['std_act_ms']:.0f} ms")
    print(f"  SLA violation: {results['sla_violation_rate']:.2%}")
    print(f"  Mean reward:   {results['mean_reward']:.4f}")

    return results


# ─────────────────────────────────────────
# Plots
# ─────────────────────────────────────────

def plot_training_curves(metrics, save_dir):
    """Save training curve plots."""
    if not metrics['episode_acts']:
        print("No episode metrics to plot yet")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Smooth with rolling average
    acts = metrics['episode_acts']
    slas = metrics['episode_sla_rates']
    window = min(50, len(acts) // 10 + 1)

    acts_smooth = pd.Series(acts).rolling(window).mean()
    slas_smooth = pd.Series(slas).rolling(window).mean()

    # ACT over training
    axes[0].plot(acts, alpha=0.2, color='steelblue')
    axes[0].plot(acts_smooth, color='steelblue', linewidth=2)
    axes[0].set_title('Average Completion Time During Training',
                       fontweight='bold')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('ACT (ms)')
    axes[0].grid(True, alpha=0.3)

    # SLA violation rate over training
    axes[1].plot(slas, alpha=0.2, color='crimson')
    axes[1].plot(slas_smooth, color='crimson', linewidth=2)
    axes[1].set_title('SLA Violation Rate During Training',
                       fontweight='bold')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('SLA Violation Rate')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('PPO Training Progress', fontsize=14, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(save_dir, 'ppo_training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved → {path}")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

if __name__ == "__main__":
    # Train PPO
    model, metrics = train_ppo(CONFIG)

    # Evaluate on all volatility conditions
    print("\n" + "=" * 60)
    print("Post-Training Evaluation")
    print("=" * 60)

    all_results = []
    for volatility in ['normal', 'high_churn', 'low_bandwidth']:
        # Twin evaluation (what agent learned)
        twin_result = evaluate_policy(
            model, volatility=volatility,
            twin_mode=True, label='PPO-twin'
        )
        all_results.append(twin_result)

        # Real network evaluation (sim-to-real)
        real_result = evaluate_policy(
            model, volatility=volatility,
            twin_mode=False, label='PPO-real'
        )
        all_results.append(real_result)

    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(
        CONFIG['results_dir'], 'ppo_evaluation.csv'
    )
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(results_df[[
        'label', 'volatility', 'mean_act_ms',
        'sla_violation_rate', 'mean_reward'
    ]].to_string(index=False))
    print(f"\nResults saved → {results_path}")
    print("\nPhase 3 complete! Next: build baselines (Phase 4)")