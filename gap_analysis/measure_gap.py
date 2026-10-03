from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "rl_agent" / "results" / "ppo_evaluation.csv"
OUT_PATH = ROOT / "gap_analysis" / "results" / "transfer_gap.csv"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing PPO evaluation file: {INPUT_PATH}")

    frame = pd.read_csv(INPUT_PATH)
    pivot = frame.pivot_table(
        index="volatility",
        columns="label",
        values=["mean_act_ms", "sla_violation_rate"],
        aggfunc="first",
    )

    required = [
        ("mean_act_ms", "PPO-twin"),
        ("mean_act_ms", "PPO-real"),
        ("sla_violation_rate", "PPO-twin"),
        ("sla_violation_rate", "PPO-real"),
    ]
    missing = [column for column in required if column not in pivot.columns]
    if missing:
        raise ValueError(f"Required PPO rows missing: {missing}")

    result = pd.DataFrame(
        {
            "volatility": pivot.index,
            "twin_act_ms": pivot[("mean_act_ms", "PPO-twin")].values,
            "real_act_ms": pivot[("mean_act_ms", "PPO-real")].values,
            "twin_sla_rate": pivot[("sla_violation_rate", "PPO-twin")].values,
            "real_sla_rate": pivot[("sla_violation_rate", "PPO-real")].values,
        }
    )

    result["act_gap_pct"] = (
        (result["real_act_ms"] - result["twin_act_ms"])
        / result["twin_act_ms"]
        * 100
    )
    result["sla_gap_pct"] = (
        (result["real_sla_rate"] - result["twin_sla_rate"])
        / result["twin_sla_rate"].replace(0, float("nan"))
        * 100
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)

    print(f"Saved transfer analysis to: {OUT_PATH}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
