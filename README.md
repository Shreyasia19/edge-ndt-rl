# AI-Driven Digital Twin Framework for Intelligent Resource Orchestration in Collaborative Edge Computing

![CI Pipeline](https://github.com/Shreyasia19/edge-ndt-rl/actions/workflows/ci.yml/badge.svg)

**B.Tech Final Year Project**

## What This Project Does

We build a Network Digital Twin (NDT) — a fast surrogate model of an edge network — to train a Reinforcement Learning (PPO) agent for microservice offloading in Collaborative Edge Computing (CEC). The trained policy is deployed onto a realistic emulated network and we explicitly measure + reduce the sim-to-real performance gap — a contribution the base paper identified as future work but never completed.

## Base Paper

Chen, X., Cao, J., Liang, Z., Sahni, Y., & Zhang, M.
*"Digital Twin-assisted Reinforcement Learning for Resource-aware Microservice Offloading in Edge Computing."*
IEEE MASS 2023, pp. 28–36.

## Team

- Shrey (Member 1) — Network Emulation + Data Pipeline
- Member 2 — Digital Twin Model (MLP + GraphSAGE)
- Member 3 — RL Agent + Baselines

## Tech Stack

| Component | Tool |
|---|---|
| Network emulation | Custom NetSim + Containernet |
| Digital twin | PyTorch (MLP) + PyTorch Geometric (GraphSAGE) |
| RL agent | Stable-Baselines3 PPO / DQN / A2C |
| ILP baseline | Google OR-Tools |
| Experiment tracking | Weights & Biases |
| Dashboard | Streamlit + FastAPI |
| CI/CD | GitHub Actions |

## Quick Setup

```bash
git clone https://github.com/Shreyasia19/edge-ndt-rl.git
cd edge-ndt-rl
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Real Experimental Results

### PPO vs All Baselines (Normal Network)

| Method | ACT (ms) | SLA Violation (%) | vs Round Robin |
|---|---|---|---|
| **PPO-NDT (ours)** | **85,299** | **95.6%** | **−72.7%** |
| Round Robin | 312,545 | 86.7% | baseline |
| Greedy | 282,952 | 87.8% | −9.5% |
| ILP (OR-Tools) | 341,039 | 91.3% | +9.1% |

### Sim-to-Real Gap (Core Contribution)

| Network Condition | Twin ACT (ms) | Real ACT (ms) | Gap % |
|---|---|---|---|
| Normal | 90,432 | 85,299 | −5.7% ✅ |
| High churn | 144,645 | 159,032 | +9.9% |
| Low bandwidth | 304,561 | 317,665 | +4.3% |

### Digital Twin Fidelity

| Condition | RMSE | R² |
|---|---|---|
| Normal | 0.1077 | 0.3101 |
| High churn | 0.1803 | 0.2115 |
| Low bandwidth | 0.1151 | 0.3095 |

## Project Status

- [x] Phase 1 — Data pipeline + network simulator (135,000 telemetry records)
- [x] Phase 2 — MLP Digital Twin (R²=0.21–0.31) + GraphSAGE twin
- [x] Phase 3 — PPO agent (10,000 episodes, 77% ACT reduction)
- [x] Phase 4 — Baselines: Round Robin, Greedy, ILP benchmarked
- [x] Phase 5a — Sim-to-real gap measured across 3 conditions
- [ ] Phase 5b — Gap reduction (domain randomization + fine-tuning)
- [ ] Phase 6 — Final dashboard + report
