"""Stage-2 configuration adapter.

Reads the repository's existing params.yaml with yaml.safe_load and exposes the
flat keys used by the training/service modules, so the stage-1 config stays intact.
"""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_params(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path is not None else REPO_ROOT / "params.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    # Already in the stage-2 shape (only if we ever ship a standalone copy).
    if "paths" in raw and "problem" in raw:
        return raw

    base = raw.get("base", {})
    data = raw.get("data", {})
    model_cfg = raw.get("model", {})
    leak = data.get("leaking_col", "waste_weighed_at_bin")
    numeric = [
        col
        for col in model_cfg.get(
            "numeric_features",
            [
                "students_registered",
                "attendance_pct",
                "menu_rating",
                "temp_c",
                "cooked_kg",
                "plate_size_cm",
            ],
        )
        if col != leak
    ]
    categorical = model_cfg.get("categorical_features", ["meal", "hostel"])
    random_state = int(base.get("random_state", 42))

    return {
        "problem": {
            "id": base.get("project_id", "MP301"),
            "group_id": base.get("group_id", "G151"),
        },
        "paths": {
            "train_csv": data.get("raw_train_path", "data/raw/train.csv"),
            "later_csv": data.get("raw_later_path", "data/raw/later.csv"),
            "holdout_csv": data.get("raw_holdout_path", "data/raw/holdout_features.csv"),
            "findings": "reports/findings.yaml",
            "model_out": "models/model.joblib",
            "predictions": "predictions/holdout_predictions.csv",
            "mlflow_db": "mlflow.db",
        },
        "data": {
            "target": data.get("target_col", model_cfg.get("target", "waste_kg")),
            "id_col": "record_id",
            "date_col": "date",
            "group_col": data.get("group_col", "hostel"),
            "categorical": categorical,
            "numeric": numeric,
            "leak_candidates": [leak],
            "sentinel_values": [-999, -1, 999, 9999],
            "unit_factors": [10, 12, 60, 100, 1000],
            "stuck_min_rows": 7,
            "stuck_min_days": 5,
        },
        "model": {
            "registered_name": "mp301_waste_model",
            "alias": "champion",
            "test_size": 0.2,
            "random_state": random_state,
            "fallback_waste_kg": 18.0,
            "candidates": [
                {"name": "ridge", "kind": "ridge", "alpha": 1.0},
                {
                    "name": "random_forest",
                    "kind": "random_forest",
                    "n_estimators": 150,
                    "max_depth": 10,
                },
                {
                    "name": "hist_gb",
                    "kind": "hist_gradient_boosting",
                    "max_iter": 250,
                    "learning_rate": 0.08,
                },
            ],
        },
        "api": {"max_cooked_kg": 1500, "min_plate_size_cm": 10},
    }


def resolve(path: str | Path, root: Path | None = None) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (root or REPO_ROOT) / candidate


def find_csv(filename: str, root: Path | None = None) -> Path:
    base = root or REPO_ROOT
    candidates = [base / "data" / "raw" / filename, base / "data" / filename, base / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {filename}. Tried: {', '.join(map(str, candidates))}")
