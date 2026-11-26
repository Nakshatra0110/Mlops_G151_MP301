"""Model factory shared by training, the service and the tests."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FallbackPredictor:
    """Tiny stand-in used by tests before model.joblib exists."""

    def __init__(self, value: float = 18.0):
        self.value = float(value)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value, dtype=float)


def build_pipeline(candidate: dict[str, Any], params: dict) -> Pipeline:
    """One preprocessor + one regressor, described entirely by params.yaml."""
    cfg = params["data"]
    numeric = [c for c in cfg["numeric"] if c not in set(cfg.get("leak_candidates", []))]
    categorical = list(cfg["categorical"])

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )

    kind = candidate["kind"]
    if kind == "ridge":
        estimator = Ridge(alpha=float(candidate.get("alpha", 1.0)))
    elif kind == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=int(candidate.get("n_estimators", 150)),
            max_depth=candidate.get("max_depth") or None,
            random_state=int(params["model"].get("random_state", 42)),
            n_jobs=-1,
        )
    elif kind == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            max_iter=int(candidate.get("max_iter", 250)),
            learning_rate=float(candidate.get("learning_rate", 0.08)),
            random_state=int(params["model"].get("random_state", 42)),
        )
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    return Pipeline([("preprocessor", preprocessor), ("regressor", estimator)])


def predict_frame(model: Any, frame: pd.DataFrame, params: dict) -> np.ndarray:
    """Predict using exactly the trained feature order."""
    cfg = params["data"]
    features = [c for c in [*cfg["numeric"], *cfg["categorical"]] if c in frame.columns]
    ordered = frame[features].copy()
    for col in cfg["numeric"]:
        if col in ordered.columns:
            ordered[col] = pd.to_numeric(ordered[col], errors="coerce")
    return np.asarray(model.predict(ordered), dtype=float).ravel()
