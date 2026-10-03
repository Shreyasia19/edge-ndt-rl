from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def artifact_path(relative_path: str) -> Path:
    return ROOT / relative_path


def read_csv(relative_path: str) -> pd.DataFrame | None:
    path = artifact_path(relative_path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def read_json(relative_path: str) -> dict:
    path = artifact_path(relative_path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def exists(relative_path: str) -> bool:
    return artifact_path(relative_path).exists()
