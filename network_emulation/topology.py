"""
topology.py
Lightweight edge network simulator for CEC environment.

Simulates heterogeneous edge nodes with:
- Dynamic CPU/memory loads
- Variable link bandwidth and latency
- Node churn (random failures and recovery)
- Task queuing per node

Motivated by: Chen et al., MASS 2023 — Section III System Model
Network model: G = (V, E), nodes V with resources AR_n,
links E with bandwidth BW_e (Equations 1-9)
"""

import numpy as np
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

np.random.seed(42)


# ─────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────

@dataclass
class EdgeNode:
    """Represents one edge node in the CEC network."""
    node_id: int
    compute_capacity_mcps: float   # max CPU (Mega cycles/sec)
    memory_gb: float               # total memory
    current_load: float = 0.0     # fraction 0-1
    current_memory_used: float = 0.0
    queue_length: int = 0          # tasks waiting
    is_online: bool = True         # churn simulation
    failure_prob: float = 0.02     # 2% chance of failure per step


@dataclass
class NetworkLink:
    """Represents a link between two edge nodes."""
    source: int
    target: int
    bandwidth_mbps: float          # max bandwidth
    latency_ms: float              # base latency
    current_bandwidth: float = 0.0  # currently used bandwidth


@dataclass
class MicroserviceTask:
    """A microservice task to be offloaded."""
    task_id: int
    service_id: int
    cpu_load_kcycles: float        # computational demand
    data_size_mbit: float          # input data size
    num_microservices: int         # DAG size
    release_time: float            # when it arrives
    deadline_ms: float             # SLA deadline
    source_node: int               # which node it came from
    assigned_node: int = -1        # -1 = unassigned
    completion_time: float = -1.0  # -1 = not done
    sla_violated: bool = False


# ─────────────────────────────────────────
# Main simulator class
# ─────────────────────────────────────────

class EdgeNetworkSimulator:
    """
    Simulates a Collaborative Edge Computing (CEC) network.

    Designed to:
    1. Generate telemetry data for training the digital twin
    2. Serve as the "real network" for sim-to-real validation
    3. Support volatility modes (normal / high-churn / low-bandwidth)

    Ref: Chen et al., MASS 2023, Section III-A System Model
    """

    VOLATILITY_MODES = {
        'normal': {
            'bandwidth_noise_std': 0.1,   # 10% noise
            'load_noise_std': 0.05,
            'failure_prob': 0.02,
            'bandwidth_scale': 1.0,
        },
        'high_churn': {
            'bandwidth_noise_std': 0.2,
            'load_noise_std': 0.1,
            'failure_prob': 0.10,          # 10% failure chance
            'bandwidth_scale': 0.8,
        },
        'low_bandwidth': {
            'bandwidth_noise_std': 0.3,
            'load_noise_std': 0.1,
            'failure_prob': 0.03,
            'bandwidth_scale': 0.3,        # 30% of normal bandwidth
        },
    }

    def __init__(
        self,
        config_path: str = "data/processed/config_8nodes.json",
        volatility: str = 'normal',
        seed: int = 42
    ):
        np.random.seed(seed)
        self.volatility = volatility
        self.vol_params = self.VOLATILITY_MODES[volatility]
        self.current_time = 0.0
        self.telemetry_log = []

        # Load config
        self._load_config(config_path)
        print(f"EdgeNetworkSimulator initialized")
        print(f"  Nodes: {len(self.nodes)}")
        print(f"  Links: {len(self.links)}")
        print(f"  Volatility mode: {volatility}")

    def _load_config(self, config_path: str):
        """Load network topology from JSON config."""
        with open(config_path) as f:
            config = json.load(f)

        self.num_nodes = config['num_nodes']

        # Build scale factor per base paper: mean=10 Mbps
        bw_scale = self.vol_params['bandwidth_scale']

        self.nodes: List[EdgeNode] = []
        for n in config['nodes']:
            self.nodes.append(EdgeNode(
                node_id=n['node_id'],
                compute_capacity_mcps=n['compute_capacity_mcps'],
                memory_gb=n['memory_gb'],
                failure_prob=self.vol_params['failure_prob'],
            ))

        self.links: List[NetworkLink] = []
        self.link_map: Dict[Tuple[int, int], NetworkLink] = {}
        for l in config['links']:
            link = NetworkLink(
                source=l['source'],
                target=l['target'],
                bandwidth_mbps=l['bandwidth_mbps'] * bw_scale,
                latency_ms=l['latency_ms'],
            )
            self.links.append(link)
            self.link_map[(l['source'], l['target'])] = link
            self.link_map[(l['target'], l['source'])] = link  # bidirectional

    # ─────────────────────────────────────────
    # State representation
    # ─────────────────────────────────────────

    def get_state(self) -> Dict:
        """
        Returns current network state — fed to both the twin and RL agent.
        Matches state space S = {CT, NI_t} from base paper Eq. 10-12.
        """
        node_features = []
        for node in self.nodes:
            node_features.append({
                'node_id': node.node_id,
                'cpu_load': round(node.current_load, 4),
                'memory_used': round(node.current_memory_used, 4),
                'queue_length': node.queue_length,
                'is_online': int(node.is_online),
                'available_capacity': round(
                    node.compute_capacity_mcps * (1 - node.current_load), 4
                ),
            })

        link_features = []
        for link in self.links:
            noise = np.random.normal(
                0, self.vol_params['bandwidth_noise_std']
            )
            effective_bw = max(
                link.bandwidth_mbps * (1 + noise) - link.current_bandwidth,
                0.1
            )
            link_features.append({
                'source': link.source,
                'target': link.target,
                'available_bandwidth_mbps': round(effective_bw, 4),
                'latency_ms': round(link.latency_ms, 4),
            })

        return {
            'time': self.current_time,
            'nodes': node_features,
            'links': link_features,
            'num_nodes': self.num_nodes,
        }

    def get_state_vector(self) -> np.ndarray:
        """
        Flat numpy array version of state — for RL agent input.
        Format: [node_load × N, node_queue × N, link_bw × L]
        """
        state = self.get_state()
        node_loads = [n['cpu_load'] for n in state['nodes']]
        node_queues = [n['queue_length'] / 100.0 for n in state['nodes']]
        node_online = [n['is_online'] for n in state['nodes']]
        link_bws = [
            l['available_bandwidth_mbps'] / 20.0
            for l in state['links']
        ]
        return np.array(
            node_loads + node_queues + node_online + link_bws,
            dtype=np.float32
        )

    def state_dim(self) -> int:
        """Returns dimension of the flat state vector."""
        return (
            3 * self.num_nodes +      # load + queue + online per node
            len(self.links)           # bandwidth per link
        )

    # ─────────────────────────────────────────
    # Task execution
    # ─────────────────────────────────────────

    def assign_task(
        self,
        task: MicroserviceTask,
        target_node_id: int,
        bandwidth_fraction: float = 0.5
    ) -> Tuple[float, bool]:
        """
        Assign a task to a target edge node and compute completion time.
        Implements Equations 1-5 from base paper.

        Returns:
            completion_time (ms), sla_violated (bool)
        """
        node = self.nodes[target_node_id]

        # Check node is online
        if not node.is_online:
            # Fallback to nearest online node
            target_node_id = self._nearest_online_node(task.source_node)
            node = self.nodes[target_node_id]

        # ── Computation cost (Eq. 1) ──────────────────────────
        # PT_n,i,j = (C_i,j / PS_n,t + W_i,j) * delta
        available_speed = node.compute_capacity_mcps * (1 - node.current_load)
        available_speed = max(available_speed, 0.1)

        # waiting time proportional to queue
        waiting_time_ms = node.queue_length * 5.0

        compute_time_ms = (
            (task.cpu_load_kcycles / 1000.0) / available_speed * 1000.0
            + waiting_time_ms
        )

        # ── Flow transmission cost (Eq. 3) ───────────────────
        # BT_f,i,j = D_i,j / B_f,i,j + epsilon_f
        link = self.link_map.get(
            (task.source_node, target_node_id)
        )
        if link:
            alloc_bandwidth = link.bandwidth_mbps * bandwidth_fraction
            alloc_bandwidth = max(alloc_bandwidth, 0.1)
            transmission_time_ms = (
                task.data_size_mbit / alloc_bandwidth * 1000.0
                + link.latency_ms
            )
        else:
            # No direct link — use average latency
            transmission_time_ms = 20.0

        # ── Total completion time (Eq. 4-5) ──────────────────
        completion_time_ms = compute_time_ms + transmission_time_ms

        # ── Update node state ─────────────────────────────────
        load_increase = min(
            task.cpu_load_kcycles / 1000.0 /
            node.compute_capacity_mcps, 0.4
        )
        node.current_load = min(node.current_load + load_increase, 1.0)
        node.current_memory_used = min(
            node.current_memory_used + 0.1, 1.0
        )
        node.queue_length += task.num_microservices

        # Update link usage
        if link:
            link.current_bandwidth = min(
                link.current_bandwidth + alloc_bandwidth,
                link.bandwidth_mbps
            )

        # ── SLA check ─────────────────────────────────────────
        sla_violated = completion_time_ms > task.deadline_ms

        # Update task
        task.assigned_node = target_node_id
        task.completion_time = completion_time_ms
        task.sla_violated = sla_violated

        return completion_time_ms, sla_violated

    # ─────────────────────────────────────────
    # Time step
    # ─────────────────────────────────────────

    def step(self, dt: float = 1.0):
        """
        Advance simulation by dt time units.
        Updates node loads, link bandwidths, and node churn.
        """
        self.current_time += dt

        for node in self.nodes:
            # Gradually recover load (tasks completing)
            recovery = np.random.uniform(0.05, 0.15)
            node.current_load = max(node.current_load - recovery, 0.0)
            node.current_memory_used = max(
                node.current_memory_used - 0.05, 0.0
            )
            node.queue_length = max(node.queue_length - 2, 0)

            # Add background load noise
            noise = np.random.normal(
                0, self.vol_params['load_noise_std']
            )
            node.current_load = np.clip(node.current_load + noise, 0, 1)

            # Node churn simulation
            if node.is_online:
                if np.random.random() < node.failure_prob:
                    node.is_online = False
            else:
                # 30% chance of recovery per step
                if np.random.random() < 0.3:
                    node.is_online = True
                    node.current_load = np.random.uniform(0, 0.3)

        # Recover link bandwidth
        for link in self.links:
            link.current_bandwidth = max(
                link.current_bandwidth - link.bandwidth_mbps * 0.2,
                0.0
            )

    def reset(self):
        """Reset simulator to initial state."""
        for node in self.nodes:
            node.current_load = np.random.uniform(0, 0.3)
            node.current_memory_used = np.random.uniform(0, 0.2)
            node.queue_length = 0
            node.is_online = True
        for link in self.links:
            link.current_bandwidth = 0.0
        self.current_time = 0.0

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────

    def _nearest_online_node(self, source: int) -> int:
        """Find nearest online node to source."""
        for i in range(self.num_nodes):
            if self.nodes[i].is_online:
                return i
        return 0  # fallback

    def online_nodes(self) -> List[int]:
        """Return list of currently online node IDs."""
        return [n.node_id for n in self.nodes if n.is_online]

    def get_node_loads(self) -> List[float]:
        """Return current load of all nodes."""
        return [n.current_load for n in self.nodes]

    def get_available_bandwidths(self) -> List[float]:
        """Return available bandwidth of all links."""
        return [
            max(l.bandwidth_mbps - l.current_bandwidth, 0)
            for l in self.links
        ]

    def summary(self) -> str:
        """Print current network status."""
        online = sum(1 for n in self.nodes if n.is_online)
        avg_load = np.mean([n.current_load for n in self.nodes])
        return (
            f"t={self.current_time:.1f} | "
            f"Online: {online}/{self.num_nodes} | "
            f"Avg load: {avg_load:.2f}"
        )


# ─────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd

    print("=" * 50)
    print("Edge Network Simulator — Quick Test")
    print("=" * 50)

    # Test all 3 volatility modes
    for mode in ['normal', 'high_churn', 'low_bandwidth']:
        print(f"\nMode: {mode}")
        sim = EdgeNetworkSimulator(
            config_path="data/processed/config_8nodes.json",
            volatility=mode
        )

        # Load a few tasks
        tasks_df = pd.read_csv("data/processed/tasks_train.csv")
        sample_tasks = tasks_df.head(5)

        completion_times = []
        sla_violations = 0

        for _, row in sample_tasks.iterrows():
            task = MicroserviceTask(
                task_id=int(row['task_id']),
                service_id=int(row['service_id']),
                cpu_load_kcycles=float(row['cpu_load_kcycles']),
                data_size_mbit=float(row['data_size_mbit']),
                num_microservices=int(row['num_microservices']),
                release_time=float(row['release_time']),
                deadline_ms=float(row['deadline_ms']),
                source_node=int(row['source_node']),
            )

            # Assign to a random online node
            online = sim.online_nodes()
            target = np.random.choice(online)
            ct, violated = sim.assign_task(task, target)
            completion_times.append(ct)
            if violated:
                sla_violations += 1

            sim.step()

        print(f"  Avg completion time: {np.mean(completion_times):.2f} ms")
        print(f"  SLA violations: {sla_violations}/5")
        print(f"  State vector dim: {sim.state_dim()}")
        print(f"  {sim.summary()}")

    print("\nSimulator test passed!")