"""
digital_twin/train_gnn_twin.py
Trains the GraphSAGE-based Network Digital Twin.

Optimized version:
  - Whole batch is flattened into ONE big disjoint graph (no Python loop
    over samples) -> far faster on CPU and GPU.
  - Auto-selects CUDA if available, otherwise CPU.
  - Dataset is kept as tensors on the device (no DataLoader overhead).
  - Epoch timing + GPU memory info.

Run: python3 -m digital_twin.train_gnn_twin
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from torch_geometric.nn import SAGEConv
    GNN_AVAILABLE = True
    print("PyTorch Geometric available ✅ — training GraphSAGE twin")
except ImportError:
    GNN_AVAILABLE = False
    print("PyTorch Geometric not available — falling back to MLP")

plt.rcParams.update({
    'figure.facecolor': '#0a0e1a', 'axes.facecolor': '#0d1b2a',
    'axes.edgecolor': '#1e3a5f', 'axes.labelcolor': '#90a4ae',
    'xtick.color': '#78909c', 'ytick.color': '#78909c',
    'text.color': '#cfd8dc', 'grid.color': '#1a2744', 'grid.alpha': .5,
})

# ── Config ────────────────────────────────────────────────────────────
CONFIG = {
    'state_dim':     52,
    'num_nodes':     8,
    'node_feat_dim': 4,     # load, queue, online, available_cap
    'edge_feat_dim': 2,
    'action_dim':    2,     # node_idx normalised, bw_fraction
    'hidden_dim':    64,
    'batch_size':    64,    # try 256 or 512 on a GPU for more speed
    'learning_rate': 0.001,
    'epochs':        150,
    'val_split':     0.2,
    'save_dir':      'digital_twin/saved_models',
    'telemetry_dir': 'data/processed/telemetry',
}
os.makedirs(CONFIG['save_dir'], exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════════════
# Graph utilities
# ══════════════════════════════════════════════════════════════════════

def _build_edge_index(num_nodes: int) -> torch.Tensor:
    """Fully connected graph — every node connects to every other."""
    src, dst = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


_EDGE_CACHE = {}


def _batched_edge_index(B: int, n: int, device) -> torch.Tensor:
    """Disjoint union of B copies of the n-node fully connected graph."""
    key = (B, n, str(device))
    if key not in _EDGE_CACHE:
        base = _build_edge_index(n).to(device)                    # [2, E]
        offsets = (torch.arange(B, device=device) * n).view(B, 1, 1)
        ei = (base.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
        _EDGE_CACHE[key] = ei
    return _EDGE_CACHE[key]


def _state_to_node_feats(state_vec: torch.Tensor,
                         num_nodes: int = 8) -> torch.Tensor:
    """Single state [state_dim] -> [N, 4] node features."""
    loads = state_vec[:num_nodes]
    queues = state_vec[num_nodes:2 * num_nodes]
    online = state_vec[2 * num_nodes:3 * num_nodes]
    avail = 1.0 - loads
    return torch.stack([loads, queues, online, avail], dim=1)


# ══════════════════════════════════════════════════════════════════════
# GraphSAGE Digital Twin
# ══════════════════════════════════════════════════════════════════════

class GraphSAGETwin(nn.Module):
    """
    GraphSAGE surrogate model.

    State layout (52 dims for 8 nodes):
      [load×8 | queue×8 | online×8 | link_bw×28]
    Node features come from the first 24 dims; graph is fully connected.
    """

    def __init__(self, node_feat_dim=4, action_dim=2,
                 hidden_dim=64, state_dim=52, num_nodes=8):
        super().__init__()
        self.num_nodes = num_nodes
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.sage1 = SAGEConv(node_feat_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_batch(self, state_batch, action_batch, num_nodes=None):
        """Vectorized forward: whole batch as one disjoint graph."""
        n = num_nodes or self.num_nodes
        B = state_batch.shape[0]

        loads = state_batch[:, :n]
        queues = state_batch[:, n:2 * n]
        online = state_batch[:, 2 * n:3 * n]
        avail = 1.0 - loads
        x = torch.stack([loads, queues, online, avail], dim=2)  # [B, n, 4]
        x = x.reshape(B * n, -1)

        edge_index = _batched_edge_index(B, n, state_batch.device)

        x = F.relu(self.bn1(self.sage1(x, edge_index)))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.bn2(self.sage2(x, edge_index)))

        graph_embed = x.view(B, n, -1).mean(dim=1)               # [B, hidden]
        combined = torch.cat([graph_embed, action_batch], dim=-1)

        next_state = self.decoder(combined)                      # [B, state_dim]
        reward = self.reward_head(combined).squeeze(-1)          # [B]
        return next_state, reward

    def forward(self, node_feats, edge_index, action):
        """Single-graph forward (kept for compatibility with other code)."""
        x = F.relu(self.bn1(self.sage1(node_feats, edge_index)))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.bn2(self.sage2(x, edge_index)))
        graph_embed = x.mean(dim=0, keepdim=True)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        combined = torch.cat([graph_embed, action], dim=-1)
        return self.decoder(combined).squeeze(0), \
            self.reward_head(combined).squeeze()


# ══════════════════════════════════════════════════════════════════════
# MLP fallback (same interface)
# ══════════════════════════════════════════════════════════════════════

class MLPTwinFallback(nn.Module):
    """Plain MLP twin — used when PyG not available."""

    def __init__(self, state_dim=52, action_dim=2, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.transition = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state, action):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        x = torch.cat([state, action], dim=-1)
        return self.transition(x).squeeze(0), self.reward_head(x).squeeze()

    def forward_batch(self, sb, ab, num_nodes=8):
        x = torch.cat([sb, ab], dim=-1)
        return self.transition(x), self.reward_head(x).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════
# Data (tensors live on the device; no DataLoader overhead)
# ══════════════════════════════════════════════════════════════════════

class TensorData:
    def __init__(self, df: pd.DataFrame, state_dim: int, device):
        sc = [f's_{i}' for i in range(state_dim)]
        nsc = [f'ns_{i}' for i in range(state_dim)]
        self.S = torch.tensor(df[sc].values, dtype=torch.float32)
        self.NS = torch.tensor(df[nsc].values, dtype=torch.float32)
        A = torch.tensor(df[['action_node', 'action_bw']].values,
                         dtype=torch.float32)
        A[:, 0] /= 8.0
        self.A = A
        self.R = torch.tensor(df['reward'].values, dtype=torch.float32)
        self.S, self.A = self.S.to(device), self.A.to(device)
        self.NS, self.R = self.NS.to(device), self.R.to(device)

    def __len__(self):
        return len(self.S)

    def batches(self, batch_size, shuffle):
        N = len(self)
        idx = (torch.randperm(N, device=self.S.device) if shuffle
               else torch.arange(N, device=self.S.device))
        for i in range(0, N, batch_size):
            b = idx[i:i + batch_size]
            if len(b) < 2:          # BatchNorm needs >1 sample
                continue
            yield self.S[b], self.A[b], self.NS[b], self.R[b]

    def num_batches(self, batch_size):
        return (len(self) + batch_size - 1) // batch_size


# ══════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════

def train(config=CONFIG):
    print("=" * 55)
    print("Network Digital Twin — GraphSAGE Training")
    print("=" * 55)
    print(f"Using device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading telemetry...")
    tel_path = os.path.join(config['telemetry_dir'], 'telemetry_all.csv')
    df = pd.read_csv(tel_path)
    print(f"Records: {len(df):,}")

    split = int(len(df) * (1 - config['val_split']))
    train_d = TensorData(df.iloc[:split], config['state_dim'], DEVICE)
    val_d = TensorData(df.iloc[split:], config['state_dim'], DEVICE)
    print(f"Train: {len(train_d):,} | Val: {len(val_d):,}")

    if GNN_AVAILABLE:
        model = GraphSAGETwin(
            node_feat_dim=config['node_feat_dim'],
            action_dim=config['action_dim'],
            hidden_dim=config['hidden_dim'],
            state_dim=config['state_dim'],
            num_nodes=config['num_nodes'],
        )
        model_name = "GraphSAGE"
    else:
        model = MLPTwinFallback(state_dim=config['state_dim'],
                                action_dim=config['action_dim'])
        model_name = "MLP (fallback)"
    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {model_name}")
    print(f"Parameters: {total_params:,}")

    bs = config['batch_size']
    opt = optim.Adam(model.parameters(), lr=config['learning_rate'])
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    loss_fn = nn.MSELoss()
    best_val = float('inf')
    train_losses, val_losses = [], []

    print(f"\nTraining {config['epochs']} epochs (batch size {bs})...")
    print("-" * 55)
    t_start = time.time()

    for epoch in range(config['epochs']):
        t_ep = time.time()

        # ── Train ────────────────────────────────────────────────────
        model.train()
        t_loss = 0.0
        nb = 0
        for S, A, NS, R in train_d.batches(bs, shuffle=True):
            opt.zero_grad()
            ns_pred, r_pred = model.forward_batch(S, A, config['num_nodes'])
            loss = loss_fn(ns_pred, NS) + 0.1 * loss_fn(r_pred.view(-1), R)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            t_loss += loss.item()
            nb += 1
        t_loss /= max(nb, 1)

        # ── Validate ──────────────────────────────────────────────────
        model.eval()
        v_loss = 0.0
        nb = 0
        with torch.no_grad():
            for S, A, NS, R in val_d.batches(max(bs, 512), shuffle=False):
                ns_pred, _ = model.forward_batch(S, A, config['num_nodes'])
                v_loss += loss_fn(ns_pred, NS).item()
                nb += 1
        v_loss /= max(nb, 1)

        sch.step(v_loss)
        train_losses.append(t_loss)
        val_losses.append(v_loss)

        if v_loss < best_val:
            best_val = v_loss
            torch.save(
                {'model_state': model.state_dict(),
                 'model_name': model_name,
                 'config': config},
                os.path.join(config['save_dir'], 'gnn_twin_best.pt')
            )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            ep_time = time.time() - t_ep
            elapsed = time.time() - t_start
            eta = ep_time * (config['epochs'] - epoch - 1)
            msg = (f"Epoch {epoch+1:3d}/{config['epochs']} | "
                   f"Train: {t_loss:.6f} | Val: {v_loss:.6f} | "
                   f"Best: {best_val:.6f} | "
                   f"{ep_time:.1f}s/ep | elapsed {elapsed/60:.1f}m | "
                   f"ETA {eta/60:.1f}m")
            if DEVICE.type == 'cuda':
                msg += f" | GPU mem {torch.cuda.max_memory_allocated()/1e6:.0f}MB"
            print(msg, flush=True)

    torch.save(
        {'model_state': model.state_dict(),
         'model_name': model_name,
         'config': config},
        os.path.join(config['save_dir'], 'gnn_twin_final.pt')
    )
    print(f"\nTotal training time: {(time.time()-t_start)/60:.1f} min")

    _compare_with_mlp(model, val_d, config)
    _plot_curves(train_losses, val_losses, model_name, config['save_dir'])

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Model: {model_name}")
    print(f"Saved to: {config['save_dir']}/gnn_twin_best.pt")
    return model, train_losses, val_losses


def _compare_with_mlp(gnn_model, val_data, config):
    """Compare GNN twin vs original MLP twin side by side."""
    try:
        from digital_twin.model import MLPDigitalTwin
    except Exception as e:
        print(f"Could not import MLPDigitalTwin ({e}) — skipping comparison")
        return

    mlp_path = os.path.join(config['save_dir'], 'twin_best.pt')
    if not os.path.exists(mlp_path):
        print("MLP twin not found — skipping comparison")
        return

    mlp = MLPDigitalTwin(
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        hidden_dim=128,
    )
    mlp.load_state_dict(torch.load(mlp_path, map_location='cpu'))
    mlp = mlp.to(DEVICE)
    mlp.eval()
    gnn_model.eval()

    loss_fn = nn.MSELoss()
    gnn_losses, mlp_losses = [], []

    with torch.no_grad():
        for S, A, NS, R in val_data.batches(512, shuffle=False):
            gnn_pred, _ = gnn_model.forward_batch(S, A, config['num_nodes'])
            gnn_losses.append(loss_fn(gnn_pred, NS).item())

            mlp_pred, _ = mlp(S, A)
            mlp_losses.append(loss_fn(mlp_pred, NS).item())

    gnn_mean = float(np.mean(gnn_losses))
    mlp_mean = float(np.mean(mlp_losses))
    improvement = (mlp_mean - gnn_mean) / mlp_mean * 100

    print("\n" + "=" * 55)
    print("GNN vs MLP Comparison")
    print("=" * 55)
    print(f"MLP val loss:       {mlp_mean:.6f}")
    print(f"GraphSAGE val loss: {gnn_mean:.6f}")
    if improvement > 0:
        print(f"GraphSAGE is {improvement:.1f}% BETTER than MLP ✅")
    else:
        print(f"MLP is {-improvement:.1f}% better than GraphSAGE")
        print("(Expected — GNN needs more data to show advantage)")

    comp = pd.DataFrame([
        {'model': 'MLP', 'val_loss': mlp_mean,
         'params': sum(p.numel() for p in mlp.parameters())},
        {'model': 'GraphSAGE', 'val_loss': gnn_mean,
         'params': sum(p.numel() for p in gnn_model.parameters())},
    ])
    comp.to_csv(os.path.join(config['save_dir'], 'gnn_vs_mlp.csv'),
                index=False)
    print(f"\nComparison saved → {config['save_dir']}/gnn_vs_mlp.csv")


def _plot_curves(train_losses, val_losses, model_name, save_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, label='Train loss', color='#4fc3f7')
    ax.plot(val_losses, label='Val loss', color='#ffa726')
    ax.set_title(f'{model_name} Digital Twin — Training Curves',
                 fontweight='bold', color='#4fc3f7')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    path = os.path.join(save_dir, 'gnn_training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves → {path}")


if __name__ == "__main__":
    train()