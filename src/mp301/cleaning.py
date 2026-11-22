"""
Data Cleaning Pipeline for MP301 Hostel Mess Food Waste.

Cleans train.csv by resolving the data issues identified during data audit:
1. Drops leaking column 'waste_weighed_at_bin'
2. Corrects unit mixup in 'plate_size_cm' for 'Boys Hostel 1' (divided by 10)
3. Imputes sentinel values (-999.0) in 'temp_c' using valid median
4. Corrects 10x typographical errors in target 'waste_kg' (> 100 kg)
5. Removes duplicate rows if any exist

Usage:
    python -m src.mp301.cleaning
"""

import os
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from .utils import load_params


def clean_dataset(
    df: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Clean an input dataframe according to the MP301 audit specifications.
    
    Args:
        df: Input DataFrame (e.g. raw train.csv).
        params: Optional dictionary of cleaning parameters. If None, loaded from params.yaml
                or defaults are used.
                
    Returns:
        Cleaned pd.DataFrame ready for modeling.
    """
    cleaned = df.copy()

    # Default parameters if not provided
    unit_col = "plate_size_cm"
    unit_group = "Boys Hostel 1"
    unit_factor = 10.0
    unit_threshold = 100.0

    sentinel_col = "temp_c"
    sentinel_val = -999.0

    target_col = "waste_kg"
    target_typo_thresh = 100.0
    target_typo_factor = 10.0

    leaking_col = "waste_weighed_at_bin"

    if params and "data" in params:
        d_cfg = params["data"]
        leaking_col = d_cfg.get("leaking_col", leaking_col)
        target_col = d_cfg.get("target_col", target_col)
        
        um_cfg = d_cfg.get("unit_mixup", {})
        unit_col = um_cfg.get("col", unit_col)
        unit_group = um_cfg.get("group_value", unit_group)
        unit_factor = float(um_cfg.get("factor", unit_factor))
        unit_threshold = float(um_cfg.get("threshold", unit_threshold))
        
        sv_cfg = d_cfg.get("sentinel_value", {})
        sentinel_col = sv_cfg.get("col", sentinel_col)
        sentinel_val = float(sv_cfg.get("value", sentinel_val))
        
        tt_cfg = d_cfg.get("target_typos", {})
        target_typo_thresh = float(tt_cfg.get("threshold", target_typo_thresh))
        target_typo_factor = float(tt_cfg.get("factor", target_typo_factor))

    # 1. Remove leaking column if present
    if leaking_col in cleaned.columns:
        cleaned = cleaned.drop(columns=[leaking_col])

    # 2. Correct unit mixup (Boys Hostel 1 recorded plate_size_cm in mm instead of cm)
    if unit_col in cleaned.columns and "hostel" in cleaned.columns:
        mask_unit = (cleaned["hostel"] == unit_group) & (cleaned[unit_col] > unit_threshold)
        cleaned.loc[mask_unit, unit_col] = cleaned.loc[mask_unit, unit_col] / unit_factor

    # 3. Handle sentinel values in temp_c (-999.0 indicates missing reading)
    if sentinel_col in cleaned.columns:
        valid_temps = cleaned.loc[cleaned[sentinel_col] != sentinel_val, sentinel_col]
        median_temp = float(valid_temps.median()) if len(valid_temps) > 0 else 26.7
        cleaned.loc[cleaned[sentinel_col] == sentinel_val, sentinel_col] = median_temp

    # 4. Correct target typos (typing mistake with extra zero: 10x too large)
    if target_col in cleaned.columns:
        mask_typo = cleaned[target_col] > target_typo_thresh
        cleaned.loc[mask_typo, target_col] = cleaned.loc[mask_typo, target_col] / target_typo_factor

    # 5. Remove duplicate rows if any
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    return cleaned


def clean_and_save(
    input_path: str = "data/raw/train.csv",
    output_path: str = "data/processed/train_cleaned.csv",
    params_path: str = "params.yaml",
) -> pd.DataFrame:
    """
    Load raw training data, apply cleaning transformations, and save cleaned CSV.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input data not found at {input_path}")

    params = None
    if os.path.exists(params_path):
        params = load_params(params_path)

    raw_df = pd.read_csv(input_path)
    cleaned_df = clean_dataset(raw_df, params=params)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved successfully to {output_path} (Shape: {cleaned_df.shape})")
    
    return cleaned_df


if __name__ == "__main__":
    clean_and_save()
