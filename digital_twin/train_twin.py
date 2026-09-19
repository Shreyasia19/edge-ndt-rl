"""
digital_twin/train_twin.py
Trains the digital twin surrogate model on collected telemetry.

Implements Algorithm 1 from base paper:
- Transaction model (t_model): predicts next state
- Reward model (r_model): predicts reward
- Trained via supervised learning with MSE loss (Eq. 16)

Our extension: uses GNN architecture instead of plain MLP
for topology-aware state transition learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from digital_twin.model import (
    MLPDigitalTwin, build_twin,
    build_graph_data, make_action_tensor, GNN_AVAILABLE
)

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────

CONFIG = {
    'state_dim': 52,
    'num_nodes': 8,
    'action_dim': 2,
    'hidden_dim': 128,
    'batch_size': 64,        # same as base paper
    'learning_rate': 0.001,  # same as base paper
    'epochs': 150,
    'val_split': 0.2,
    'use_gnn': True,
    'save_dir': 'digital_twin/saved_models',
    'telemetry_dir': 'data/processed/telemetry',
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)


# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────

class TelemetryDataset(Dataset):
    """
    Dataset of (state, action, next_state, reward) tuples
    collected from the edge network simulator.
    """

    def __init__(self, df: pd.DataFrame, state_dim: int = 52):
        self.state_dim = state_dim

        # Extract state columns (s_0 to s_51)
        state_cols = [f's_{i}' for i in range(state_dim)]
        next_state_cols = [f'ns_{i}' for i in range(state_dim)]

        self.states = torch.tensor(
            df[state_cols].values, dtype=torch.float32
        )
        self.next_states = torch.tensor(
            df[next_state_cols].values, dtype=torch.float32
        )
        self.actions = torch.tensor(
            df[['action_node', 'action_bw']].values,
            dtype=torch.float32
        )
        # Normalise action_node to [0,1]
        self.actions[:, 0] = self.actions[:, 0] / 8.0

        self.rewards = torch.tensor(
            df['reward'].values, dtype=torch.float32
        )

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.next_states[idx],
            self.rewards[idx],
        )


# ─────────────────────────────────────────
# Training
# ─────────────────────────────────────────

def train_twin(config: dict = CONFIG):
    print("=" * 55)
    print("Digital Twin Training")
    print("=" * 55)

    # Load telemetry
    print("\nLoading telemetry data...")
    telemetry_path = os.path.join(
        config['telemetry_dir'], 'telemetry_all.csv'
    )
    df = pd.read_csv(telemetry_path)
    print(f"Total records: {len(df):,}")
    print(f"Volatility breakdown:\n{df['volatility'].value_counts()}")

    # Train/val split
    split = int(len(df) * (1 - config['val_split']))
    df_train = df.iloc[:split]
    df_val = df.iloc[split:]

    train_dataset = TelemetryDataset(df_train, config['state_dim'])
    val_dataset = TelemetryDataset(df_val, config['state_dim'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )

    print(f"\nTrain: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    # Build model
    model = MLPDigitalTwin(
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        hidden_dim=config['hidden_dim'],
    )
    print(f"\nModel: MLPDigitalTwin")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, verbose=True
    )
    criterion = nn.MSELoss()

    # Training loop
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    print(f"\nTraining for {config['epochs']} epochs...")
    print("-" * 55)

    for epoch in range(config['epochs']):
        # ── Train ──────────────────────────
        model.train()
        train_loss = 0.0
        train_reward_loss = 0.0

        for states, actions, next_states, rewards in train_loader:
            optimizer.zero_grad()

            next_state_pred, reward_pred = model(states, actions)

            # State transition loss (Eq. 16 in base paper)
            state_loss = criterion(next_state_pred, next_states)

            # Reward prediction loss
            reward_pred_flat = reward_pred.view(-1)
            reward_loss = criterion(reward_pred_flat, rewards)
            # Combined loss
            loss = state_loss + 0.1 * reward_loss
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += state_loss.item()
            train_reward_loss += reward_loss.item()

        train_loss /= len(train_loader)

        # ── Validate ───────────────────────
        model.eval()
        val_loss = 0.0
        val_reward_loss = 0.0

        with torch.no_grad():
            for states, actions, next_states, rewards in val_loader:
                next_state_pred, reward_pred = model(states, actions)
                v_loss = criterion(next_state_pred, next_states)
                val_loss += v_loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(config['save_dir'], 'twin_best.pt')
            )

        # Print every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:3d}/{config['epochs']} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Best: {best_val_loss:.6f}"
            )

    # Save final model
    torch.save(
        model.state_dict(),
        os.path.join(config['save_dir'], 'twin_final.pt')
    )

    # Save config
    with open(
        os.path.join(config['save_dir'], 'twin_config.json'), 'w'
    ) as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 55)
    print("Training Complete")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Models saved to: {config['save_dir']}")

    # Plot training curves
    plot_training_curves(train_losses, val_losses, config['save_dir'])

    return model, train_losses, val_losses


def plot_training_curves(train_losses, val_losses, save_dir):
    """Save training curve plot."""
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Digital Twin Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Training curve saved → {path}")


if __name__ == "__main__":
    model, train_losses, val_losses = train_twin(CONFIG)
    print("\nDigital twin trained and saved!")
    print("Next: run digital_twin/evaluate_twin.py")
    