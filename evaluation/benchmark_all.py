from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.greedy import evaluate_greedy
from baselines.ilp_baseline import evaluate_ilp
from baselines.round_robin import evaluate_round_robin

SCENARIOS = ["normal", "high_churn", "low_bandwidth"]
OUT_PATH = ROOT / "evaluation" / "results" / "benchmark.csv"
PPO_PATH = ROOT / "rl_agent" / "results" / "ppo_evaluation.csv"


def standardise(row: dict, method: str, source: str, episodes: int, tasks: int) -> dict:
    return {
        "method": method,
        "volatility": row["volatility"],
        "mean_act_ms": row["mean_act_ms"],
        "std_act_ms": row["std_act_ms"],
        "sla_violation_rate": row["sla_violation_rate"],
        "n_episodes": episodes,
        "tasks_per_episode": tasks,
        "avg_solve_time_ms": row.get("avg_solve_time_ms"),
        "source": source,
    }


def load_existing_ppo() -> list[dict]:
    if not PPO_PATH.exists():
        print("PPO result CSV not found; PPO rows will be skipped.")
        return []

    frame = pd.read_csv(PPO_PATH)
    frame = frame[frame["label"] == "PPO-real"]

    rows = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "method": "PPO",
                "volatility": row["volatility"],
                "mean_act_ms": row["mean_act_ms"],
                "std_act_ms": row["std_act_ms"],
                "sla_violation_rate": row["sla_violation_rate"],
                "n_episodes": row.get("n_episodes", 50),
                "tasks_per_episode": 20,
                "avg_solve_time_ms": None,
                "source": "saved trained-policy evaluation",
            }
        )
    return rows


def main() -> None:
    results = load_existing_ppo()

    for scenario in SCENARIOS:
        print(f"\nRunning baselines for: {scenario}")

        results.append(
            standardise(
                evaluate_round_robin(
                    volatility=scenario,
                    n_episodes=50,
                    tasks_per_episode=20,
                ),
                "Round Robin",
                "fresh evaluation",
                50,
                20,
            )
        )

        results.append(
            standardise(
                evaluate_greedy(
                    volatility=scenario,
                    n_episodes=50,
                    tasks_per_episode=20,
                ),
                "Greedy",
                "fresh evaluation",
                50,
                20,
            )
        )

        # ILP remains a small-snapshot lower bound. It is labelled clearly in CSV.
        results.append(
            standardise(
                evaluate_ilp(
                    volatility=scenario,
                    n_episodes=20,
                    tasks_per_episode=15,
                ),
                "ILP lower bound",
                "small-snapshot evaluation",
                20,
                15,
            )
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    frame.to_csv(OUT_PATH, index=False)

    print(f"\nSaved benchmark results to: {OUT_PATH}")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
