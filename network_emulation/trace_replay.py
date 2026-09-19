"""
trace_replay.py
Preprocesses Alibaba Cluster Trace v2018 into task arrival format
compatible with our edge network emulation.

Based on: Chen et al., MASS 2023 (Section V-A Real-world Dataset)
- Computational load: scaled from (end_time - start_time) * plan_cpu
- Data size: Gaussian, mean=500 Mbit, std=80%
- Release time: Poisson distribution
"""

import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm

np.random.seed(42)  # reproducibility

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_alibaba_batch_task(filepath):
    """Load and clean the batch_task CSV from Alibaba trace."""
    print(f"Loading {filepath}...")
    
    # Alibaba batch_task columns:
    # task_name, inst_num, status, start_time, end_time, plan_cpu, plan_mem
    df = pd.read_csv(
        filepath,
        header=None,
        names=['task_name', 'inst_num', 'status',
               'start_time', 'end_time', 'plan_cpu', 'plan_mem'],
        nrows=50000  # use first 50k rows — enough for our project
    )
    
    print(f"Loaded {len(df)} rows")
    
    # Keep only completed tasks
    df = df[df['status'] == 'Terminated'].copy()
    print(f"After filtering terminated tasks: {len(df)} rows")
    
    # Drop rows with missing values
    df = df.dropna(subset=['start_time', 'end_time', 'plan_cpu'])
    
    return df


def generate_tasks_from_alibaba(df, num_nodes=8):
    """
    Convert Alibaba trace into our task format.
    Matches base paper's dataset construction (Section V-A).
    """
    tasks = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing tasks"):
        # Computational load: scale from CPU * duration (base paper method)
        duration = max(float(row['end_time']) - float(row['start_time']), 1)
        cpu_load = float(row['plan_cpu']) * duration
        # Scale to kilo clock cycles (base paper: mean=500 kcycles, std=60%)
        cpu_load_kcycles = max(cpu_load * 500, 10)
        
        # Data size: Gaussian mean=500 Mbit, std=80% (exactly as base paper)
        data_size_mbit = max(np.random.normal(500, 400), 10)
        
        # Number of microservices per service: Gaussian within [1, 50]
        num_microservices = int(np.clip(np.random.normal(10, 8), 1, 50))
        
        # Release time: Poisson distribution (base paper)
        release_time = np.random.poisson(lam=float(row['start_time']) % 1000)
        
        # Source edge node: randomly assigned
        source_node = np.random.randint(0, num_nodes)
        
        tasks.append({
            'task_id': idx,
            'service_id': idx // 5,  # group ~5 tasks per service
            'cpu_load_kcycles': round(cpu_load_kcycles, 2),
            'data_size_mbit': round(data_size_mbit, 2),
            'num_microservices': num_microservices,
            'release_time': release_time,
            'source_node': source_node,
            'deadline_ms': round(cpu_load_kcycles * 0.05 + 200, 2),  # SLA deadline
        })
    
    return pd.DataFrame(tasks)


def generate_synthetic_tasks(num_tasks=5000, num_nodes=8):
    """
    Fallback: generate synthetic tasks using exact base paper distributions.
    Use this if Alibaba download fails.
    Ref: Chen et al., MASS 2023, Section V-A Synthetic Dataset
    """
    print("Generating synthetic tasks using base paper distributions...")
    tasks = []
    
    for i in range(num_tasks):
        # Number of microservices: Gaussian within [1, 50]
        num_microservices = int(np.clip(np.random.normal(10, 8), 1, 50))
        
        # Computational load: Gaussian mean=500 kcycles, std=60%
        cpu_load = max(np.random.normal(500, 300), 10)
        
        # Data size: Gaussian mean=500 Mbit, std=80%
        data_size = max(np.random.normal(500, 400), 10)
        
        # Release time: Poisson
        release_time = np.random.poisson(lam=i * 0.5)
        
        # Source node: random
        source_node = np.random.randint(0, num_nodes)
        
        tasks.append({
            'task_id': i,
            'service_id': i // 5,
            'cpu_load_kcycles': round(cpu_load, 2),
            'data_size_mbit': round(data_size, 2),
            'num_microservices': num_microservices,
            'release_time': release_time,
            'source_node': source_node,
            'deadline_ms': round(cpu_load * 0.05 + 200, 2),
        })
    
    return pd.DataFrame(tasks)


def generate_network_config(num_nodes=8):
    """
    Generate edge network configuration.
    Matches base paper: 8 nodes, Gaussian compute capacity,
    Gaussian link bandwidth (Section V-A Network Model).
    """
    np.random.seed(99)  # fixed seed for reproducible topology
    nodes = []
    for i in range(num_nodes):
        # Compute capacity: Gaussian mean=40 Mcps, std=80% (base paper exact)
        compute_capacity = max(np.random.normal(40, 32), 5)
        # Memory: Gaussian mean=4GB, std=50%
        memory_gb = max(np.random.normal(4, 2), 0.5)
        
        nodes.append({
            'node_id': i,
            'compute_capacity_mcps': round(compute_capacity, 2),
            'memory_gb': round(memory_gb, 2),
        })
    
    links = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Bandwidth: Gaussian mean=10 Mbps, std=80% (base paper exact)
            bandwidth = max(np.random.normal(10, 8), 0.5)
            # Latency: Gaussian mean=5ms, std=50%
            latency = max(np.random.normal(5, 2.5), 0.1)
            
            links.append({
                'source': i,
                'target': j,
                'bandwidth_mbps': round(bandwidth, 2),
                'latency_ms': round(latency, 2),
            })
    
    config = {
        'num_nodes': num_nodes,
        'nodes': nodes,
        'links': links,
        'description': 'Base config: 8 nodes, matching Chen et al. MASS 2023'
    }
    
    return config


def save_network_configs():
    """Save multiple network configs for scalability testing."""
    configs = {
        'config_8nodes.json': 8,    # same as base paper
        'config_20nodes.json': 20,  # our scalability extension
        'config_50nodes.json': 50,  # large-scale test
    }
    
    for filename, num_nodes in configs.items():
        config = generate_network_config(num_nodes)
        path = os.path.join(PROCESSED_DIR, filename)
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved {filename} ({num_nodes} nodes)")


if __name__ == "__main__":
    print("=" * 50)
    print("Alibaba Cluster Trace Preprocessing")
    print("=" * 50)
    
    # Try real Alibaba data first
    alibaba_file = os.path.join(RAW_DIR, "batch_task.csv")
    
    if os.path.exists(alibaba_file):
        print("Found Alibaba trace file — using real data")
        df_raw = load_alibaba_batch_task(alibaba_file)
        df_tasks = generate_tasks_from_alibaba(df_raw)
    else:
        print("Alibaba trace not found — using synthetic data")
        print("(This is valid per base paper's synthetic dataset methodology)")
        df_tasks = generate_synthetic_tasks(num_tasks=5000)
    
    # Split into train/test (80/20) — same as base paper
    split = int(len(df_tasks) * 0.8)
    df_train = df_tasks[:split]
    df_test = df_tasks[split:]
    
    # Save
    df_train.to_csv(os.path.join(PROCESSED_DIR, "tasks_train.csv"), index=False)
    df_test.to_csv(os.path.join(PROCESSED_DIR, "tasks_test.csv"), index=False)
    df_tasks.to_csv(os.path.join(PROCESSED_DIR, "tasks_all.csv"), index=False)
    
    print(f"\nSaved:")
    print(f"  tasks_train.csv — {len(df_train)} tasks")
    print(f"  tasks_test.csv  — {len(df_test)} tasks")
    print(f"  tasks_all.csv   — {len(df_tasks)} tasks")
    
    # Generate network configs
    print("\nGenerating network configs...")
    save_network_configs()
    
    # Print sample
    print("\nSample tasks:")
    print(df_tasks.head())
    print("\nData stats:")
    print(df_tasks.describe())
    
    print("\nPreprocessing complete!")