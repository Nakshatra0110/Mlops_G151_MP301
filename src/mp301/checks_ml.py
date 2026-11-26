"""Seven data-quality checks for MP301.

Each function returns plain Python dictionaries (JSON/YAML serialisable) so the
same code can be used by the training script and demonstrated live in the viva.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include=[np.number]).columns)


def _text_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _numeric_columns(df)]


def _clean_numeric(series: pd.Series, sentinels: list[int | float]) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.mask(values.isin(sentinels))


def check_leaking_column(
    train: pd.DataFrame,
    holdout: pd.DataFrame | None,
    params: dict,
) -> dict:
    """A column absent from holdout that almost exactly restates the target."""
    cfg = params["data"]
    target = cfg["target"]
    id_col = cfg["id_col"]
    date_col = cfg["date_col"]
    holdout_cols = set(holdout.columns) if holdout is not None else set()
    hinted = set(cfg.get("leak_candidates", []))

    y = pd.to_numeric(train[target], errors="coerce").astype(float)
    # The typo check is independent of this one; correcting it first stops a
    # handful of 10x targets from hiding the near-perfect leak agreement.
    typo = check_target_typos(train, params)
    if typo["present"]:
        too_big = y > float(typo["threshold_kg"])
        y = y.mask(too_big, y[too_big] / float(typo["factor"]))
    candidates: list[dict] = []
    for col in train.columns:
        if col in {target, id_col, date_col}:
            continue
        absent_from_holdout = holdout_cols and col not in holdout_cols
        hinted_column = col in hinted
        if not (absent_from_holdout or hinted_column):
            continue
        x = pd.to_numeric(train[col], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 10:
            continue
        corr = float(x[mask].corr(y[mask]))
        abs_err = float((x[mask] - y[mask]).abs().mean())
        rel_err = abs_err / max(float(y[mask].abs().mean()), 1e-9)
        candidates.append(
            {
                "column": col,
                "absent_from_holdout": bool(absent_from_holdout),
                "corr_with_target": round(corr, 6),
                "mae_vs_target_kg": round(abs_err, 6),
                "relative_mae": round(rel_err, 6),
            }
        )

    candidates.sort(key=lambda item: (item["relative_mae"], -abs(item["corr_with_target"])))
    best = candidates[0] if candidates else None
    present = bool(best and (best["relative_mae"] < 0.15 or abs(best["corr_with_target"]) > 0.9))
    return {
        "present": present,
        "column": best["column"] if present and best else None,
        "details": best if present and best else {},
        "all_candidates": candidates,
    }


def check_unit_mixup(train: pd.DataFrame, params: dict) -> dict:
    """One hostel recorded a numeric column in another (multiplicative) unit."""
    cfg = params["data"]
    group_col = cfg["group_col"]
    target = cfg["target"]
    sentinels = cfg.get("sentinel_values", [])
    factors = cfg.get("unit_factors", [10, 12, 60, 100, 1000])
    ignore = {target, cfg["id_col"], cfg["date_col"], *cfg.get("leak_candidates", [])}

    best: dict | None = None
    for col in _numeric_columns(train):
        if col in ignore:
            continue
        values = _clean_numeric(train[col], sentinels)
        work = pd.DataFrame({"value": values, "group": train[group_col].astype(str)}).dropna()
        if work.empty:
            continue
        for group in sorted(work["group"].unique()):
            inside = work.loc[work["group"] == group, "value"]
            outside = work.loc[work["group"] != group, "value"]
            if len(inside) < 20 or len(outside) < 20:
                continue
            inside_med, outside_med = float(inside.median()), float(outside.median())
            if abs(inside_med) < 1e-9 or abs(outside_med) < 1e-9:
                continue
            ratio = inside_med / outside_med
            for factor in factors:
                divide = ratio > 1
                scale = float(factor) if divide else 1.0 / float(factor)
                corrected_med = inside_med / scale if divide else inside_med * scale
                median_error = abs(corrected_med - outside_med) / abs(outside_med)
                inside_scaled = inside / scale if divide else inside * scale
                iqr_error = abs(float(inside_scaled.quantile(0.75)) - float(outside.quantile(0.75))) / max(
                    abs(float(outside.quantile(0.75))), 1e-9
                )
                score = 0.7 * median_error + 0.3 * iqr_error
                if score > 0.25:
                    continue
                candidate = {
                    "column": col,
                    "group_column": group_col,
                    "group": group,
                    "factor": int(factor),
                    "direction": "divide" if divide else "multiply",
                    "observed_median": round(inside_med, 4),
                    "peer_median": round(outside_med, 4),
                    "corrected_median": round(corrected_med, 4),
                    "relative_error": round(score, 6),
                }
                if best is None or score < best["relative_error"]:
                    best = candidate

    present = best is not None
    return {
        "present": present,
        "column": best["column"] if present else None,
        "group_column": group_col,
        "group": best["group"] if present else None,
        "factor": best["factor"] if present else None,
        "direction": best["direction"] if present else None,
        "details": best if present else {},
    }


def check_sentinel_values(train: pd.DataFrame, params: dict) -> dict:
    """Repeated impossible numbers such as -1 or 9999."""
    sentinels = params["data"].get("sentinel_values", [])
    min_count = max(3, int(0.004 * len(train)))
    found: list[dict] = []
    for col in _numeric_columns(train):
        counts = train[col].value_counts(dropna=True)
        for sentinel in sentinels:
            count = int(counts.get(sentinel, 0))
            if count >= min_count:
                found.append(
                    {
                        "column": col,
                        "value": sentinel,
                        "count": count,
                        "share": round(count / len(train), 6),
                    }
                )
    found.sort(key=lambda item: -item["count"])
    return {
        "present": bool(found),
        "count_of_affected_cells": int(sum(item["count"] for item in found)),
        "details": found,
    }


def check_duplicate_rows(train: pd.DataFrame) -> dict:
    extra = int(train.duplicated().sum())
    return {
        "present": extra > 0,
        "extra_copies": extra,
        "rows_after_drop": int(len(train) - extra),
    }


def check_stuck_sensor(train: pd.DataFrame, params: dict) -> dict:
    """A numeric column frozen on the same reading across consecutive days."""
    cfg = params["data"]
    date_col, group_col = cfg["date_col"], cfg["group_col"]
    min_rows = int(cfg.get("stuck_min_rows", 7))
    min_days = int(cfg.get("stuck_min_days", 5))
    ignore = {cfg["target"], cfg["id_col"], date_col, *cfg.get("leak_candidates", [])}
    work = train.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col]).sort_values(date_col)
    longest: dict | None = None

    for col in _numeric_columns(work):
        if col in ignore:
            continue
        run_id = (work[col] != work[col].shift()).cumsum()
        for (_, _), block in work.groupby([work[group_col].astype(str), run_id], sort=False):
            block = block[block[col].notna()]
            if len(block) < min_rows:
                continue
            span_days = int((block[date_col].max() - block[date_col].min()).days)
            if span_days < min_days:
                continue
            candidate = {
                "column": col,
                "group_column": group_col,
                "group": str(block[group_col].iloc[0]),
                "stuck_value": float(block[col].iloc[0]),
                "first_date": block[date_col].min().strftime("%Y-%m-%d"),
                "last_date": block[date_col].max().strftime("%Y-%m-%d"),
                "rows": int(len(block)),
                "days": span_days,
            }
            if longest is None or (candidate["rows"], candidate["days"]) > (
                longest["rows"],
                longest["days"],
            ):
                longest = candidate

    return {
        "present": longest is not None,
        "details": longest if longest else {},
    }


def check_target_typos(train: pd.DataFrame, params: dict) -> dict:
    """A few targets typed with an extra zero (about 10x the normal range)."""
    target = params["data"]["target"]
    y = pd.to_numeric(train[target], errors="coerce").dropna().sort_values()
    if len(y) < 10:
        return {"present": False, "factor": 10, "details": {}}
    values = y.to_numpy(dtype=float)
    positive = values[values > 0]
    if len(positive) < 10:
        return {"present": False, "factor": 10, "details": {}}
    ratios = positive[1:] / np.maximum(positive[:-1], 1e-9)
    median = float(np.median(positive))
    best_index, best_ratio = None, 0.0
    for index, ratio in enumerate(ratios):
        if index < len(ratios) * 0.7:
            continue
        if positive[index + 1] > 4 * median and ratio > best_ratio:
            best_index, best_ratio = index, float(ratio)
    first_typo = positive[best_index + 1] if best_index is not None else np.inf
    if best_index is None or best_ratio < 2.5 or first_typo < 3 * median:
        return {"present": False, "factor": 10, "details": {}}
    typo_values = positive[best_index + 1 :]
    return {
        "present": True,
        "factor": 10,
        "count": int(len(typo_values)),
        "gap_ratio": round(best_ratio, 4),
        "smallest_typo": round(float(typo_values.min()), 4),
        "largest_typo": round(float(typo_values.max()), 4),
        "threshold_kg": round(float(positive[best_index]), 4),
        "details": {
            "values": [round(float(v), 4) for v in typo_values[:25]],
        },
    }


def _normalise_text(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[.\-_]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def check_spelling_variants(train: pd.DataFrame, later: pd.DataFrame | None, params: dict) -> dict:
    """One text column contains several spellings of the same category."""
    text_cols = [c for c in _text_columns(train) if c not in {params["data"]["date_col"]}]
    clean_levels: dict[str, set[str]] = {}
    if later is not None:
        for col in text_cols:
            if col in later.columns:
                clean_levels[col] = set(later[col].dropna().astype(str).str.strip().unique())

    best: dict | None = None
    all_findings: list[dict] = []
    for col in text_cols:
        groups: dict[str, set[str]] = defaultdict(set)
        for raw in train[col].dropna().astype(str).unique():
            groups[_normalise_text(raw)].add(raw)
        variants = {key: sorted(values) for key, values in groups.items() if len(values) > 1}
        if not variants:
            continue
        canonical_source = clean_levels.get(col, set())
        canonical_by_key: dict[str, str] = {}
        for key, raws in variants.items():
            matches = [level for level in canonical_source if _normalise_text(level) == key]
            canonical_by_key[key] = matches[0] if matches else min(raws, key=len)
        mapping = {
            raw: canonical_by_key[key]
            for key, raws in variants.items()
            for raw in raws
        }
        finding = {
            "column": col,
            "variants": variants,
            "canonical_mapping": mapping,
            "true_levels": len(canonical_source) if canonical_source else len(groups),
            "train_raw_levels": int(train[col].nunique(dropna=True)),
        }
        all_findings.append(finding)
        if best is None or len(mapping) > len(best["canonical_mapping"]):
            best = finding

    return {
        "present": best is not None,
        "column": best["column"] if best else None,
        "true_levels": best["true_levels"] if best else None,
        "details": best if best else {},
        "all_text_columns": all_findings,
    }


def run_all_checks(
    train: pd.DataFrame,
    later: pd.DataFrame | None,
    holdout: pd.DataFrame | None,
    params: dict,
) -> dict:
    """Run all seven checks and return one findings dictionary."""
    return {
        "leaking_column": check_leaking_column(train, holdout, params),
        "unit_mixup": check_unit_mixup(train, params),
        "sentinel_value": check_sentinel_values(train, params),
        "duplicate_rows": check_duplicate_rows(train),
        "stuck_sensor": check_stuck_sensor(train, params),
        "target_typos": check_target_typos(train, params),
        "spelling_variants": check_spelling_variants(train, later, params),
    }
