"""
Unit tests for MP301 Data Cleaning & Data Audit Pipeline.

Note: As per Student Handbook guidelines, tests do not depend on raw CSVs.
They build small synthetic test data inside the tests.
"""

import numpy as np
import pandas as pd
import pytest

from src.mp301.cleaning import clean_dataset
from src.mp301.data_checks import (
    check_duplicate_rows,
    check_leaking_column,
    check_sentinel_values,
    check_target_typos,
    check_unit_mixup,
)
from src.mp301.utils import load_params


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Creates a small representative synthetic dataset with MP301 data issues."""
    return pd.DataFrame({
        "record_id": [f"R{i:05d}" for i in range(1, 11)],
        "date": ["2026-01-01"] * 10,
        "students_registered": [500] * 10,
        "attendance_pct": [75] * 10,
        "menu_rating": [3.5] * 10,
        "temp_c": [25.0, 26.0, -999.0, 24.0, 27.0, -999.0, 28.0, 25.5, 26.5, 27.5],
        "cooked_kg": [300.0] * 10,
        "plate_size_cm": [270.0, 260.0, 280.0, 25.0, 26.0, 27.0, 28.0, 25.5, 26.5, 27.5],
        "meal": ["lunch"] * 10,
        "hostel": [
            "Boys Hostel 1", "Boys Hostel 1", "Boys Hostel 1",
            "Boys Hostel 2", "Boys Hostel 2", "Girls Hostel",
            "Girls Hostel", "Girls Hostel", "Girls Hostel", "Girls Hostel"
        ],
        "waste_weighed_at_bin": [25.1, 24.9, 30.2, 27.8, 32.1, 39.8, 22.2, 28.9, 31.2, 25.8],
        "waste_kg": [25.0, 250.0, 30.0, 28.0, 32.0, 400.0, 22.0, 29.0, 31.0, 26.0],
    })


def test_check_leaking_column(sample_raw_data):
    """Test detection of leaking column absent from holdout."""
    holdout_mock = sample_raw_data.drop(columns=["waste_weighed_at_bin", "waste_kg"])
    res = check_leaking_column(sample_raw_data, holdout_mock)
    assert res["present"] is True
    assert res["column"] == "waste_weighed_at_bin"


def test_check_unit_mixup(sample_raw_data):
    """Test detection of 10x unit mixup in plate_size_cm for Boys Hostel 1."""
    res = check_unit_mixup(sample_raw_data, group_col="hostel")
    assert res["present"] is True
    assert res["column"] == "plate_size_cm"
    assert res["group_value"] == "Boys Hostel 1"
    assert res["factor"] == 10


def test_check_sentinel_values(sample_raw_data):
    """Test detection of sentinel value -999 in temp_c."""
    res = check_sentinel_values(sample_raw_data)
    assert res["present"] is True
    assert res["column"] == "temp_c"
    assert res["value"] == -999
    assert res["count"] == 2


def test_check_target_typos(sample_raw_data):
    """Test detection of 10x target typos in waste_kg."""
    res = check_target_typos(sample_raw_data)
    assert res["present"] is True
    assert res["count"] == 2


def test_check_duplicate_rows(sample_raw_data):
    """Test duplicate rows detection."""
    res = check_duplicate_rows(sample_raw_data)
    assert res["present"] is False
    assert res["count"] is None

    # Add duplicate row
    df_dup = pd.concat([sample_raw_data, sample_raw_data.iloc[[0]]], ignore_index=True)
    res_dup = check_duplicate_rows(df_dup)
    assert res_dup["present"] is True
    assert res_dup["count"] == 1


def test_clean_dataset_removes_leakage(sample_raw_data):
    """Cleaned dataset must not have leaking column waste_weighed_at_bin."""
    cleaned = clean_dataset(sample_raw_data)
    assert "waste_weighed_at_bin" not in cleaned.columns


def test_clean_dataset_fixes_unit_mixup(sample_raw_data):
    """Cleaned dataset must scale Boys Hostel 1 plate size to cm range (< 50 cm)."""
    cleaned = clean_dataset(sample_raw_data)
    bh1_plates = cleaned[cleaned["hostel"] == "Boys Hostel 1"]["plate_size_cm"]
    assert (bh1_plates < 50.0).all()
    assert (bh1_plates >= 10.0).all()


def test_clean_dataset_imputes_sentinel(sample_raw_data):
    """Cleaned dataset must replace -999 sentinel in temp_c with realistic temperatures."""
    cleaned = clean_dataset(sample_raw_data)
    assert -999.0 not in cleaned["temp_c"].values
    assert (cleaned["temp_c"] > 0).all()
    assert (cleaned["temp_c"] < 50).all()


def test_clean_dataset_fixes_target_typos(sample_raw_data):
    """Cleaned dataset must correct target typos > 100 kg by dividing by 10."""
    cleaned = clean_dataset(sample_raw_data)
    assert (cleaned["waste_kg"] < 100.0).all()
    # Check originally 250.0 became 25.0 and 400.0 became 40.0
    assert cleaned.loc[1, "waste_kg"] == 25.0
    assert cleaned.loc[5, "waste_kg"] == 40.0


def test_load_params():
    """Verify params.yaml can be safely loaded and has required keys."""
    params = load_params("params.yaml")
    assert "data" in params
    assert "unit_mixup" in params["data"]
    assert "sentinel_value" in params["data"]
    assert "target_typos" in params["data"]
