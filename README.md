# AI-Driven Digital Twin Framework for Intelligent Resource Orchestration in Collaborative Edge Computing

**B.Tech Final Year Project**

## What This Project Does
We build a Network Digital Twin (NDT) to cheaply train a Reinforcement Learning agent
for microservice offloading in Collaborative Edge Computing (CEC). The trained policy
is deployed onto a realistic emulated network and we measure + reduce the sim-to-real
performance gap — a contribution the base paper explicitly identified as future work.

## Base Paper
Chen, X., Cao, J., Liang, Z., Sahni, Y., & Zhang, M.
"Digital Twin-assisted Reinforcement Learning for Resource-aware Microservice
Offloading in Edge Computing." IEEE MASS 2023, pp. 28-36.

## Team
- Member 1 (Shrey) — Network Emulation
- Member 2 — Digital Twin Model  
- Member 3 — RL Agent + Baselines

## Tech Stack
| Component | Tool |
|---|---|
| Network emulation | Containernet + K3s |
| Digital twin | PyTorch Geometric (GNN) |
| RL agent | Stable-Baselines3 PPO/DQN |
| ILP baseline | Google OR-Tools |
| Experiment tracking | Weights & Biases |
| Dashboard | Streamlit |

## Quick Setup
```bash
git clone https://github.com/Shreyasia19/edge-ndt-rl.git
cd edge-ndt-rl
pip install -r requirements.txt
```

## Project Status
- [x] Phase 1 — Environment Setup
- [ ] Phase 2 — Digital Twin
- [ ] Phase 3 — RL Agent
- [ ] Phase 4 — Baselines
- [ ] Phase 5 — Sim-to-Real Gap Analysis
- [ ] Phase 6 — Evaluation + Dashboard

## Results (updated as project progresses)
| Method | ACT (ms) | SLA Violation (%) | Gap (%) |
|---|---|---|---|
| Round Robin | - | - | - |
| Greedy | - | - | - |
| ILP | - | - | - |
| Our NDT+RL (twin) | - | - | - |
| Our NDT+RL (real, before) | - | - | - |
| Our NDT+RL (real, after) | - | - | - |
