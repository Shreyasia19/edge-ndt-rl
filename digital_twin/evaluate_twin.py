"""
digital_twin/evaluate_twin.py
Evaluates digital twin fidelity — measures how accurately
the twin predicts real network state transitions.

This is a key result section of the project:
- RMSE per state feature
- Predicted vs actual distribution plots
- Fidelity score per volatility condition
- Identifies WHERE the twin is inaccurate (motivates the gap)

The "twin fidelity" result directly explains WHY a sim-to-real
gap exists when we deploy the RL agent on the real network.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from digital_twin.model import MLPDigitalTwin

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────

EVAL_CONFIG = {
    'state_dim': 52,
    'num_nodes': 8,
    'model_path': 'digital_twin/saved_models/twin_best.pt',
    'telemetry_dir': 'data/processed/telemetry',
    'results_dir': 'digital_twin/saved_models/eval_results',
}

os.makedirs(EVAL_CONFIG['results_dir'], exist_ok=True)

# Human-readable feature names for plots
def get_feature_names(num_nodes=8, num_links=28):
    names = []
    names += [f'node_{i}_load' for i in range(num_nodes)]
    names += [f'node_{i}_queue' for i in range(num_nodes)]
    names += [f'node_{i}_online' for i in range(num_nodes)]
    names += [f'link_{i}_bw' for i in range(num_links)]
    return names


# ─────────────────────────────────────────
# Load model
# ─────────────────────────────────────────

def load_model(config):
    model = MLPDigitalTwin(
        state_dim=config['state_dim'],
        action_dim=2,
        hidden_dim=128,
    )
    model.load_state_dict(
        torch.load(config['model_path'], map_location='cpu')
    )
    model.eval()
    print(f"Model loaded from {config['model_path']}")
    return model


# ─────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────

def evaluate_fidelity(model, df, state_dim=52, label="all"):
    """
    Compute twin fidelity metrics on a telemetry dataframe.

    Returns:
        dict with RMSE, MAE, per-feature RMSE, predictions
    """
    state_cols = [f's_{i}' for i in range(state_dim)]
    next_state_cols = [f'ns_{i}' for i in range(state_dim)]

    states = torch.tensor(
        df[state_cols].values, dtype=torch.float32
    )
    actions = torch.tensor(
        df[['action_node', 'action_bw']].values,
        dtype=torch.float32
    )
    actions[:, 0] /= 8.0  # normalise node index

    next_states_true = df[next_state_cols].values

    # Run predictions in batches
    predictions = []
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(states), batch_size):
            s_batch = states[i:i+batch_size]
            a_batch = actions[i:i+batch_size]
            pred, _ = model(s_batch, a_batch)
            predictions.append(pred.numpy())

    predictions = np.vstack(predictions)

    # Overall RMSE and MAE
    errors = predictions - next_states_true
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))

    # Per-feature RMSE
    per_feature_rmse = np.sqrt(np.mean(errors ** 2, axis=0))

    # R² score
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum(
        (next_states_true - next_states_true.mean(axis=0)) ** 2
    )
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    print(f"\n[{label}] Twin Fidelity Metrics:")
    print(f"  RMSE:  {rmse:.6f}")
    print(f"  MAE:   {mae:.6f}")
    print(f"  R²:    {r2:.4f}")

    return {
        'label': label,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'per_feature_rmse': per_feature_rmse,
        'predictions': predictions,
        'ground_truth': next_states_true,
    }


# ─────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────

def plot_per_feature_rmse(results_by_mode, feature_names, save_dir):
    """
    Bar chart of RMSE per feature group across volatility modes.
    This is your main 'twin fidelity' result plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'normal': 'steelblue',
              'high_churn': 'darkorange',
              'low_bandwidth': 'crimson'}

    num_nodes = 8
    groups = {
        'Node Load': (0, num_nodes),
        'Node Queue': (num_nodes, 2*num_nodes),
        'Node Online': (2*num_nodes, 3*num_nodes),
        'Link BW': (3*num_nodes, len(feature_names)),
    }

    for ax, (mode, result) in zip(axes, results_by_mode.items()):
        group_rmse = {}
        for group_name, (start, end) in groups.items():
            group_rmse[group_name] = np.mean(
                result['per_feature_rmse'][start:end]
            )

        bars = ax.bar(
            group_rmse.keys(),
            group_rmse.values(),
            color=colors.get(mode, 'gray'),
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5
        )
        ax.set_title(f'Volatility: {mode}', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSE')
        ax.set_xlabel('Feature Group')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, group_rmse.values()):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.0001,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=9
            )

    plt.suptitle(
        'Digital Twin Fidelity — RMSE by Feature Group',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    path = os.path.join(save_dir, 'fidelity_per_feature.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {path}")


def plot_predicted_vs_actual(result, save_dir, mode='normal'):
    """
    Scatter plot: predicted vs actual for key features.
    Points on the diagonal = perfect prediction.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Pick 4 representative features to plot
    feature_indices = {
        'Node 0 Load': 0,
        'Node 1 Load': 1,
        'Node 0 Queue': 8,
        'Link 0 BW': 24,
    }

    preds = result['predictions']
    truth = result['ground_truth']

    for ax, (name, idx) in zip(axes, feature_indices.items()):
        p = preds[:500, idx]   # sample 500 points
        t = truth[:500, idx]

        ax.scatter(t, p, alpha=0.3, s=10, color='steelblue')

        # Perfect prediction line
        lim = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lim, lim, 'r--', linewidth=1.5, label='Perfect')

        rmse = np.sqrt(np.mean((p - t) ** 2))
        ax.set_title(f'{name}\nRMSE={rmse:.4f}', fontsize=11)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'Twin: Predicted vs Actual — {mode}',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    path = os.path.join(save_dir, f'pred_vs_actual_{mode}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {path}")


def plot_fidelity_summary(results_by_mode, save_dir):
    """
    Summary bar chart comparing RMSE/R² across volatility modes.
    This goes directly into your project report.
    """
    modes = list(results_by_mode.keys())
    rmses = [results_by_mode[m]['rmse'] for m in modes]
    r2s = [results_by_mode[m]['r2'] for m in modes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ['steelblue', 'darkorange', 'crimson']

    # RMSE
    bars = ax1.bar(modes, rmses, color=colors, alpha=0.8,
                   edgecolor='black')
    ax1.set_title('Twin Fidelity — RMSE by Volatility Mode',
                  fontweight='bold')
    ax1.set_ylabel('RMSE (lower is better)')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, rmses):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.0001,
                 f'{val:.4f}', ha='center', va='bottom')

    # R²
    bars2 = ax2.bar(modes, r2s, color=colors, alpha=0.8,
                    edgecolor='black')
    ax2.set_title('Twin Fidelity — R² by Volatility Mode',
                  fontweight='bold')
    ax2.set_ylabel('R² Score (higher is better)')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, r2s):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f'{val:.4f}', ha='center', va='bottom')

    plt.suptitle('Digital Twin Fidelity Summary',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'fidelity_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {path}")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("Digital Twin Fidelity Evaluation")
    print("=" * 55)

    # Load model
    model = load_model(EVAL_CONFIG)
    feature_names = get_feature_names()

    # Evaluate per volatility mode
    results_by_mode = {}
    summary_rows = []

    for mode in ['normal', 'high_churn', 'low_bandwidth']:
        path = os.path.join(
            EVAL_CONFIG['telemetry_dir'],
            f'telemetry_{mode}.csv'
        )
        df = pd.read_csv(path)

        result = evaluate_fidelity(
            model, df,
            state_dim=EVAL_CONFIG['state_dim'],
            label=mode
        )
        results_by_mode[mode] = result

        # Predicted vs actual scatter plots
        plot_predicted_vs_actual(
            result,
            EVAL_CONFIG['results_dir'],
            mode=mode
        )

        summary_rows.append({
            'volatility': mode,
            'rmse': round(result['rmse'], 6),
            'mae': round(result['mae'], 6),
            'r2': round(result['r2'], 4),
        })

    # Summary plots
    plot_per_feature_rmse(
        results_by_mode, feature_names, EVAL_CONFIG['results_dir']
    )
    plot_fidelity_summary(
        results_by_mode, EVAL_CONFIG['results_dir']
    )

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(
        EVAL_CONFIG['results_dir'], 'fidelity_summary.csv'
    )
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 55)
    print("FIDELITY SUMMARY TABLE")
    print("=" * 55)
    print(summary_df.to_string(index=False))

    print(f"\nAll plots saved to: {EVAL_CONFIG['results_dir']}")
    print("\nKey insight:")
    for row in summary_rows:
        gap_risk = "LOW" if row['r2'] > 0.7 else \
                   "MEDIUM" if row['r2'] > 0.4 else "HIGH"
        print(
            f"  {row['volatility']:15s} R²={row['r2']:.4f} "
            f"→ Sim-to-real gap risk: {gap_risk}"
        )
    print("\nEvaluation complete!")