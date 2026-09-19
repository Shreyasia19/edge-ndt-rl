"""
digital_twin/model.py
GNN-based Network Digital Twin surrogate model.

Learns to predict next network state given (current_state, action).
This is the "State Transition Model" from base paper Eq. 15-16,
extended with graph structure (GNN) instead of a plain MLP.

Base paper Eq. 15: s_hat_{t+1} = f_phi_s(s_t, a_t)
Base paper Eq. 16: min MSE between predicted and actual next state

Our extension: f_phi_s is a GraphSAGE network that encodes
topology-aware node and link features — an improvement over
the base paper's flat neural network transition model.

Fallback: if PyTorch Geometric is unavailable, we use a plain
MLP (MLPTwin) which is still a valid surrogate model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Try to import PyTorch Geometric — fall back to MLP if unavailable
try:
    from torch_geometric.nn import SAGEConv
    from torch_geometric.data import Data
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    print("PyTorch Geometric not available — using MLP twin (fallback)")


# ─────────────────────────────────────────
# GNN-based Digital Twin (primary)
# ─────────────────────────────────────────

class GNNDigitalTwin(nn.Module):
    """
    Graph Neural Network surrogate model for the edge network.

    Architecture:
    - GraphSAGE encoder: processes node features with topology
    - MLP decoder: predicts next state from encoded graph + action

    Input:
        node_features: [N, node_feat_dim] — per-node state
        edge_index: [2, E] — graph connectivity
        edge_features: [E, edge_feat_dim] — per-link bandwidth/latency
        action: [action_dim] — offloading decision + bandwidth

    Output:
        next_state: [state_dim] — predicted next flat state vector
        reward_pred: [1] — predicted reward (for reward shaping)
    """

    def __init__(
        self,
        node_feat_dim: int = 4,    # cpu_load, mem_used, queue, is_online
        edge_feat_dim: int = 2,    # bandwidth, latency
        action_dim: int = 2,       # target_node (normalised), bw_fraction
        hidden_dim: int = 64,
        state_dim: int = 52,       # flat state vector size
    ):
        super().__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.action_dim = action_dim
        self.state_dim = state_dim

        # GraphSAGE layers — topology-aware message passing
        self.sage1 = SAGEConv(node_feat_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)

        # Edge feature encoder
        self.edge_encoder = nn.Linear(edge_feat_dim, hidden_dim)

        # Combine graph embedding + action → predict next state
        graph_embed_dim = hidden_dim  # mean pooled over nodes
        self.decoder = nn.Sequential(
            nn.Linear(graph_embed_dim + action_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),   # next state prediction
        )

        # Reward predictor head (Eq. 17 in base paper)
        self.reward_head = nn.Sequential(
            nn.Linear(graph_embed_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_feats, edge_index, action):
        """
        Forward pass.

        Args:
            node_feats: [N, node_feat_dim]
            edge_index: [2, E]
            action: [batch, action_dim] or [action_dim]

        Returns:
            next_state_pred: [state_dim]
            reward_pred: [1]
        """
        # GraphSAGE encoding
        x = F.relu(self.sage1(node_feats, edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.sage2(x, edge_index))

        # Global mean pooling — single graph embedding vector
        graph_embed = x.mean(dim=0, keepdim=True)  # [1, hidden_dim]

        # Ensure action has batch dimension
        if action.dim() == 1:
            action = action.unsqueeze(0)

        # Concatenate graph embedding + action
        combined = torch.cat([graph_embed, action], dim=-1)

        # Predict next state and reward
        next_state = self.decoder(combined).squeeze(0)
        reward = self.reward_head(combined).squeeze()

        return next_state, reward


# ─────────────────────────────────────────
# MLP fallback twin
# ─────────────────────────────────────────

class MLPDigitalTwin(nn.Module):
    """
    MLP-based surrogate model — fallback when PyG unavailable.
    Same interface as GNNDigitalTwin.
    Matches the base paper's original MLP transition model (Eq. 15-16).
    """

    def __init__(
        self,
        state_dim: int = 52,
        action_dim: int = 2,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        input_dim = state_dim + action_dim

        self.transition_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        self.reward_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state, action):
        """
        Args:
            state: [state_dim]
            action: [action_dim]
        Returns:
            next_state: [state_dim]
            reward: [1]
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        x = torch.cat([state, action], dim=-1)
        next_state = self.transition_net(x).squeeze(0)
        reward = self.reward_head(x).squeeze()

        return next_state, reward


# ─────────────────────────────────────────
# Data preparation helpers
# ─────────────────────────────────────────

def build_graph_data(state_vector: np.ndarray, num_nodes: int = 8):
    """
    Convert flat state vector into PyG graph Data object.

    State vector layout (from topology.py get_state_vector):
    [node_loads × N | node_queues × N | node_online × N | link_bws × L]
    """
    # Extract node features from state vector
    loads = state_vector[:num_nodes]
    queues = state_vector[num_nodes:2*num_nodes]
    online = state_vector[2*num_nodes:3*num_nodes]

    # Node feature matrix: [N, 4]
    # cpu_load, queue_norm, is_online, available (1-load)
    node_feats = np.column_stack([
        loads,
        queues,
        online,
        1.0 - loads,  # available capacity
    ])
    node_feats = torch.tensor(node_feats, dtype=torch.float32)

    # Build fully connected edge index for 8 nodes
    sources, targets = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                sources.append(i)
                targets.append(j)
    edge_index = torch.tensor(
        [sources, targets], dtype=torch.long
    )

    return node_feats, edge_index


def make_action_tensor(action_node: int, action_bw: float,
                       num_nodes: int = 8) -> torch.Tensor:
    """Convert discrete action to normalised action tensor."""
    return torch.tensor([
        action_node / num_nodes,   # normalise node index to [0,1]
        action_bw,                  # already in [0,1]
    ], dtype=torch.float32)


# ─────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────

def build_twin(state_dim: int = 52, use_gnn: bool = True):
    """
    Build and return the digital twin model.
    Automatically falls back to MLP if GNN unavailable.
    """
    if use_gnn and GNN_AVAILABLE:
        print("Building GNN Digital Twin (PyTorch Geometric)")
        model = GNNDigitalTwin(
            node_feat_dim=4,
            edge_feat_dim=2,
            action_dim=2,
            hidden_dim=64,
            state_dim=state_dim,
        )
    else:
        print("Building MLP Digital Twin (fallback)")
        model = MLPDigitalTwin(
            state_dim=state_dim,
            action_dim=2,
            hidden_dim=128,
        )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    return model


# ─────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("Digital Twin Model — Quick Test")
    print("=" * 50)

    state_dim = 52
    num_nodes = 8

    # Test MLP twin (always works)
    print("\n1. Testing MLP Twin...")
    mlp_twin = MLPDigitalTwin(state_dim=state_dim)
    dummy_state = torch.randn(state_dim)
    dummy_action = torch.tensor([0.5, 0.7])
    next_s, reward = mlp_twin(dummy_state, dummy_action)
    print(f"   Input state shape: {dummy_state.shape}")
    print(f"   Predicted next state shape: {next_s.shape}")
    print(f"   Predicted reward: {reward.item():.4f}")
    print("   MLP Twin test passed ✅")

    # Test GNN twin if available
    if GNN_AVAILABLE:
        print("\n2. Testing GNN Twin...")
        gnn_twin = GNNDigitalTwin(state_dim=state_dim)
        dummy_state_np = np.random.rand(state_dim).astype(np.float32)
        node_feats, edge_index = build_graph_data(
            dummy_state_np, num_nodes
        )
        action = make_action_tensor(3, 0.6, num_nodes)
        next_s, reward = gnn_twin(node_feats, edge_index, action)
        print(f"   Node features shape: {node_feats.shape}")
        print(f"   Edge index shape: {edge_index.shape}")
        print(f"   Predicted next state shape: {next_s.shape}")
        print(f"   Predicted reward: {reward.item():.4f}")
        print("   GNN Twin test passed ✅")
    else:
        print("\n2. GNN Twin skipped (PyG not installed)")
        print("   Install: pip3 install torch-geometric")

    print("\nModel test complete!")