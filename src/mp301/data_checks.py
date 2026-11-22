"""
The Seven Data Checks for MP301 Hostel Mess Food Waste.

This module automates the detection and verification of the 7 data problems
described in the Student Handbook (Section 7) for MP301:
1. leaking_column
2. unit_mixup
3. sentinel_value
4. duplicate_rows
5. stuck_sensor
6. target_typos
7. spelling_variants

Can be executed standalone during faculty viva:
    python -m src.mp301.data_checks
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


def check_leaking_column(
    train_df: pd.DataFrame,
    holdout_df: Optional[pd.DataFrame] = None,
    target_col: str = "waste_kg",
) -> Dict[str, Any]:
    """
    Check for columns filled in after the event that secretly leak the target.
    """
    result: Dict[str, Any] = {"present": False, "column": None}
    
    # Columns in train but missing from holdout (excluding target)
    if holdout_df is not None:
        missing_in_holdout = [
            c for c in train_df.columns
            if c not in holdout_df.columns and c != target_col
        ]
    else:
        missing_in_holdout = [c for c in train_df.columns if "bin" in c.lower() or "weighed" in c.lower()]

    for col in missing_in_holdout:
        if col in train_df.columns:
            # Check correlation or relationship with target
            corr = train_df[col].corr(train_df[target_col])
            # If target has 10x typos, correlation after dividing >100 is > 0.99
            temp_target = train_df[target_col].copy()
            temp_target[temp_target > 100] /= 10.0
            corr_corrected = train_df[col].corr(temp_target)
            
            # If standard deviation is 0 or correlation is NaN, check if values closely match
            close_match = np.nanmean(np.abs(train_df[col] - temp_target)) < 5.0

            if abs(corr) > 0.3 or abs(corr_corrected) > 0.95 or close_match:
                result["present"] = True
                result["column"] = col
                result["correlation_with_target"] = float(corr) if not np.isnan(corr) else None
                result["correlation_with_clean_target"] = float(corr_corrected) if not np.isnan(corr_corrected) else None
                break

    return result


def check_unit_mixup(
    train_df: pd.DataFrame,
    group_col: str = "hostel",
) -> Dict[str, Any]:
    """
    Check if one group recorded a column in a different unit (e.g., mm vs cm).
    """
    result: Dict[str, Any] = {
        "present": False,
        "column": None,
        "group_value": None,
        "factor": None,
    }

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ["waste_kg", "waste_weighed_at_bin"]:
            continue
        # Group stats
        grouped = train_df.groupby(group_col)[col].agg(["mean", "median", "min", "max"])
        medians = grouped["median"].to_dict()
        
        # Compare ratios between groups
        group_names = list(medians.keys())
        for g1 in group_names:
            other_medians = [medians[g2] for g2 in group_names if g2 != g1 and medians[g2] > 0]
            if not other_medians:
                continue
            avg_other = float(np.mean(other_medians))
            ratio = medians[g1] / avg_other if avg_other > 0 else 1.0
            
            # Check common factor unit mixups: 10, 12, 60, 100, 1000
            for candidate_factor in [10, 12, 60, 100, 1000]:
                if 0.85 * candidate_factor <= ratio <= 1.15 * candidate_factor:
                    result["present"] = True
                    result["column"] = col
                    result["group_value"] = g1
                    result["factor"] = int(candidate_factor)
                    return result

    return result


def check_sentinel_values(
    train_df: pd.DataFrame,
    threshold_count: int = 1,
) -> Dict[str, Any]:
    """
    Check for special numbers typed instead of blanks (-1, -999, 999, 9999).
    """
    result: Dict[str, Any] = {
        "present": False,
        "column": None,
        "value": None,
        "count": None,
    }

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    common_sentinels = [-999, -999.0, 999, 9999, -1, -99]
    
    for col in numeric_cols:
        vc = train_df[col].value_counts()
        for s in common_sentinels:
            if s in vc.index:
                cnt = int(vc[s])
                # Sentinel must be an impossible or heavily repeated outlier
                if cnt >= threshold_count:
                    result["present"] = True
                    result["column"] = col
                    result["value"] = int(s) if float(s).is_integer() else float(s)
                    result["count"] = cnt
                    return result

    return result


def check_duplicate_rows(
    train_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Check for duplicate rows in train.csv.
    """
    extra_copies = int(train_df.duplicated().sum())
    return {
        "present": extra_copies > 0,
        "count": extra_copies if extra_copies > 0 else None,
    }


def check_stuck_sensor(
    train_df: pd.DataFrame,
    date_col: str = "date",
    threshold_days: int = 14,
) -> Dict[str, Any]:
    """
    Check if a sensor was stuck sending identical values for an extended run.
    """
    result: Dict[str, Any] = {
        "present": False,
        "column": None,
        "start_date": None,
        "end_date": None,
    }

    if date_col not in train_df.columns:
        return result

    df_sorted = train_df.sort_values(date_col).reset_index(drop=True)
    numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        s = df_sorted[col]
        # Ignore sentinel values when checking for stuck sensor
        s_clean = s.replace([-999, -999.0], np.nan)
        runs = (s_clean != s_clean.shift()).cumsum()
        counts = s_clean.groupby(runs).transform("count")
        max_run = counts.max()
        
        # Consider stuck if same continuous reading for threshold_days or more
        if max_run >= threshold_days:
            idx = counts[counts == max_run].index
            result["present"] = True
            result["column"] = col
            result["start_date"] = str(df_sorted.loc[idx[0], date_col])
            result["end_date"] = str(df_sorted.loc[idx[-1], date_col])
            return result

    return result


def check_target_typos(
    train_df: pd.DataFrame,
    target_col: str = "waste_kg",
    threshold: float = 100.0,
) -> Dict[str, Any]:
    """
    Check if a small group of target values were typed 10x too big.
    """
    result: Dict[str, Any] = {
        "present": False,
        "count": None,
    }

    if target_col not in train_df.columns:
        return result

    typos = train_df[train_df[target_col] > threshold]
    typo_count = len(typos)

    # In MP301: normal values max ~68.2, typos sit between 236 and 587
    if 0 < typo_count < len(train_df) * 0.5:
        corrected = typos[target_col] / 10.0
        normal_subset = train_df[train_df[target_col] <= threshold][target_col]
        normal_range_max = normal_subset.max() if len(normal_subset) > 0 else 100.0
        if (corrected <= max(normal_range_max * 1.5, 100.0)).all():
            result["present"] = True
            result["count"] = typo_count

    return result


def check_spelling_variants(
    train_df: pd.DataFrame,
    later_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Check for category spelling variants/typos in text columns.
    """
    result: Dict[str, Any] = {
        "present": False,
        "column": None,
        "true_levels": None,
    }

    text_cols = train_df.select_dtypes(include=["object", "string"]).columns
    ignore_cols = ["record_id", "date"]

    for col in text_cols:
        if col in ignore_cols:
            continue
        train_vals = train_df[col].dropna().unique()
        if later_df is not None and col in later_df.columns:
            later_vals = later_df[col].dropna().unique()
            # If train has more messy variants than later
            if len(train_vals) > len(later_vals):
                result["present"] = True
                result["column"] = col
                result["true_levels"] = len(later_vals)
                return result

    return result


def run_all_checks(
    train_path: str = "data/raw/train.csv",
    later_path: Optional[str] = "data/raw/later.csv",
    holdout_path: Optional[str] = "data/raw/holdout_features.csv",
) -> Dict[str, Any]:
    """
    Run all 7 checks on train.csv and return findings matching findings.yaml.
    """
    train_df = pd.read_csv(train_path)
    later_df = pd.read_csv(later_path) if later_path else None
    holdout_df = pd.read_csv(holdout_path) if holdout_path else None

    findings = {
        "leaking_column": check_leaking_column(train_df, holdout_df),
        "unit_mixup": check_unit_mixup(train_df),
        "sentinel_value": check_sentinel_values(train_df),
        "duplicate_rows": check_duplicate_rows(train_df),
        "stuck_sensor": check_stuck_sensor(train_df),
        "target_typos": check_target_typos(train_df),
        "spelling_variants": check_spelling_variants(train_df, later_df),
    }

    return findings


if __name__ == "__main__":
    import json
    print("=" * 70)
    print("  MP301 --- SEVEN DATA PROBLEMS AUDIT (Group G151)")
    print("=" * 70)
    
    findings = run_all_checks()
    
    for issue_name, details in findings.items():
        present = details.get("present", False)
        status = "[PRESENT]" if present else "[NOT PRESENT]"
        print(f"\n{status:<15} {issue_name}")
        for k, v in details.items():
            if k != "present":
                print(f"   - {k}: {v}")
                
    print("\n" + "=" * 70)
    print("Summary for findings.yaml:")
    print(json.dumps(findings, indent=2))
