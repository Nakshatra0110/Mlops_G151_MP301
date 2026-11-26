"""Deterministic cleaning built directly from the seven checks."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


def _normalise(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[.\-_]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def replace_sentinels(df: pd.DataFrame, checks: dict, params: dict) -> pd.DataFrame:
    out = df.copy()
    for item in checks["sentinel_value"]["details"]:
        col, value = item["column"], item["value"]
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").replace(value, np.nan)
    return out


def fix_unit_mixup(df: pd.DataFrame, checks: dict, params: dict) -> pd.DataFrame:
    detail = checks["unit_mixup"]["details"]
    if not detail:
        return df.copy()
    out = df.copy()
    col, group_col = detail["column"], detail["group_column"]
    group, factor = str(detail["group"]), float(detail["factor"])
    mask = out[group_col].astype(str) == group
    values = pd.to_numeric(out.loc[mask, col], errors="coerce")
    if detail["direction"] == "divide":
        out.loc[mask, col] = values / factor
    else:
        out.loc[mask, col] = values * factor
    return out


def fix_stuck_sensor(df: pd.DataFrame, checks: dict, params: dict) -> pd.DataFrame:
    detail = checks["stuck_sensor"]["details"]
    if not detail:
        return df.copy()
    out = df.copy()
    col, group_col, date_col = detail["column"], detail["group_column"], params["data"]["date_col"]
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    mask = (
        (out[group_col].astype(str) == str(detail["group"]))
        & (out[date_col] >= pd.Timestamp(detail["first_date"]))
        & (out[date_col] <= pd.Timestamp(detail["last_date"]))
    )
    out.loc[mask, col] = np.nan
    return out


def fix_target_typos(df: pd.DataFrame, checks: dict, params: dict) -> pd.DataFrame:
    detail = checks["target_typos"]
    if not detail["present"]:
        return df.copy()
    out = df.copy()
    target = params["data"]["target"]
    threshold = float(detail["threshold_kg"])
    factor = float(detail["factor"])
    values = pd.to_numeric(out[target], errors="coerce")
    out.loc[values > threshold, target] = values[values > threshold] / factor
    return out


def normalise_categories(
    df: pd.DataFrame,
    checks: dict | None = None,
    params: dict | None = None,
    reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Map train spelling variants onto the clean levels seen in later.csv."""
    out = df.copy()
    categorical = (params or {}).get("data", {}).get("categorical", [])
    checks = checks or {}
    all_text = checks.get("spelling_variants", {}).get("all_text_columns", [])

    for col in categorical:
        if col not in out.columns:
            continue
        mapping: dict[str, str] = {}
        for finding in all_text:
            if finding["column"] == col:
                mapping.update(finding.get("canonical_mapping", {}))
        if reference is not None and col in reference.columns:
            canonical = {_normalise(level): str(level) for level in reference[col].dropna().unique()}
            for raw in out[col].dropna().astype(str).unique():
                key = _normalise(raw)
                if key in canonical:
                    mapping[raw] = canonical[key]
        if mapping:
            out[col] = out[col].astype(str).replace(mapping)
    return out


def clean_train(
    train: pd.DataFrame,
    checks: dict,
    params: dict,
    later: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply every detected fix in a stable order."""
    out = drop_duplicate_rows(train)
    out = replace_sentinels(out, checks, params)
    out = fix_unit_mixup(out, checks, params)
    out = fix_stuck_sensor(out, checks, params)
    out = normalise_categories(out, checks, params, reference=later)
    out = fix_target_typos(out, checks, params)
    return out.reset_index(drop=True)


def prepare_xy(df: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features only: no id, no date, no target, no leak column."""
    cfg = params["data"]
    leaky = set(cfg.get("leak_candidates", []))
    if "leaking_column" in cfg:
        leaky.add(cfg["leaking_column"])
    features = [
        col
        for col in [*cfg["numeric"], *cfg["categorical"]]
        if col in df.columns and col not in leaky
    ]
    y = pd.to_numeric(df[cfg["target"]], errors="coerce")
    return df[features].copy(), y
