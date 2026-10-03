from __future__ import annotations

from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

from rl_agent.train_agent import CONFIG, evaluate_policy

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "rl_agent" / "policies" / "best_model.zip"
OUT_PATH = ROOT / "rl_agent" / "results" / "ppo_evaluation.csv"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing trained PPO policy: {MODEL_PATH}")

    model = PPO.load(str(MODEL_PATH))
    rows = []

    for volatility in ["normal", "high_churn", "low_bandwidth"]:
        rows.append(
            evaluate_policy(
                model,
                volatility=volatility,
                twin_mode=True,
                label="PPO-twin",
                n_episodes=50,
            )
        )
        rows.append(
            evaluate_policy(
                model,
                volatility=volatility,
                twin_mode=False,
                label="PPO-real",
                n_episodes=50,
            )
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"Saved PPO evaluation to: {OUT_PATH}")


if __name__ == "__main__":
    main()
