"""Tests use small in-memory frames; no CSVs are needed in CI."""
import numpy as np
import pandas as pd
import yaml

from src.mp301.cleaning_ml import clean_train, normalise_categories, prepare_xy
from src.mp301.checks_ml import run_all_checks
from src.mp301.config_ml import REPO_ROOT, load_params


def tiny_params():
    params = load_params()
    params["data"]["stuck_min_rows"] = 3
    params["data"]["stuck_min_days"] = 2
    return params


def tiny_frame():
    rows = []
    dates = pd.date_range("2026-01-01", periods=12)
    for i, date in enumerate(dates):
        rows.append(
            {
                "record_id": f"r{i}",
                "date": str(date.date()),
                "students_registered": 500,
                "attendance_pct": -1 if i in (2, 4, 6) else 70,
                "menu_rating": 4 if i != 3 else 4.0,
                "temp_c": 20,
                "cooked_kg": 200,
                "plate_size_cm": 20,
                "meal": "lunch" if i % 2 else " Lunch ",
                "hostel": "Boys Hostel 1",
                "waste_weighed_at_bin": 18,
                "waste_kg": 180 if i == 10 else 18,
            }
        )
    frame = pd.DataFrame(rows)
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    return frame


def test_all_seven_checks_return_serialisable_results():
    params = tiny_params()
    frame = tiny_frame()
    later = frame.drop(columns=["waste_weighed_at_bin", "waste_kg"]).copy()
    later["meal"] = "lunch"
    checks = run_all_checks(frame, later, later, params)
    assert len(checks) == 7
    assert checks["duplicate_rows"]["extra_copies"] == 1
    assert checks["sentinel_value"]["present"] is True
    assert checks["leaking_column"]["present"] is True
    assert checks["target_typos"]["present"] is True


def test_cleaning_removes_duplicates_sentinels_and_typos():
    params = tiny_params()
    frame = tiny_frame()
    checks = run_all_checks(frame, None, None, params)
    clean = clean_train(frame, checks, params)
    assert len(clean) == len(frame) - 1
    assert clean["attendance_pct"].isna().sum() == 3
    assert clean["waste_kg"].max() < 100
    assert clean["waste_kg"].iloc[10] == 18.0


def test_prepare_xy_never_includes_the_leaking_column():
    params = tiny_params()
    frame = tiny_frame()
    X, y = prepare_xy(frame, params)
    assert "waste_weighed_at_bin" not in X.columns
    assert "waste_kg" not in X.columns
    assert "record_id" not in X.columns
    assert "date" not in X.columns
    assert len(X) == len(y)


def test_normalise_categories_uses_later_clean_spellings():
    params = tiny_params()
    frame = tiny_frame()
    canonical = pd.DataFrame({"meal": ["lunch", "breakfast"], "hostel": ["Boys Hostel 1", "Boys Hostel 1"]})
    checks = run_all_checks(frame, canonical, None, params)
    clean = normalise_categories(frame, checks, params, reference=canonical)
    assert set(clean["meal"].unique()) == {"lunch"}
    assert np.isfinite(clean["temp_c"].to_numpy()).all()


"""Focused tests for the unit-mixup and stuck-sensor detectors."""
import pandas as pd
import yaml

from src.mp301.cleaning_ml import fix_unit_mixup
from src.mp301.config_ml import REPO_ROOT, load_params
from src.mp301.checks_ml import check_stuck_sensor, check_unit_mixup


def params():
    return load_params()


def frame_with_unit_mixup():
    rows = []
    for i in range(60):
        rows.append(
            {
                "record_id": f"r{i}",
                "date": f"2026-03-{(i % 28) + 1:02d}",
                "students_registered": 500,
                "attendance_pct": 70,
                "menu_rating": 4,
                "temp_c": 20,
                "cooked_kg": 200,
                "plate_size_cm": 20 if i % 2 == 0 else 200,
                "meal": "lunch",
                "hostel": "Boys Hostel 1" if i % 2 == 0 else "Girls Hostel",
                "waste_kg": 18,
            }
        )
    return pd.DataFrame(rows)


def test_unit_mixup_finds_exact_factor_and_direction():
    cfg = params()
    frame = frame_with_unit_mixup()
    finding = check_unit_mixup(frame, cfg)
    assert finding["present"] is True
    assert finding["column"] == "plate_size_cm"
    assert finding["factor"] == 10
    assert finding["direction"] == "divide"
    fixed = fix_unit_mixup(frame, {"unit_mixup": finding}, cfg)
    assert round(fixed.loc[fixed["hostel"] == "Girls Hostel", "plate_size_cm"].median(), 3) == 20.0


def test_stuck_sensor_reports_first_and_last_date():
    cfg = params()
    cfg["data"]["stuck_min_rows"] = 5
    cfg["data"]["stuck_min_days"] = 4
    rows = []
    for i in range(20):
        date = pd.Timestamp("2026-04-01") + pd.Timedelta(days=i)
        rows.append(
            {
                "record_id": f"r{i}",
                "date": str(date.date()),
                "students_registered": 500 + i,
                "attendance_pct": 70 + (i % 3),
                "menu_rating": 4 + (i % 2),
                "temp_c": 31 if 5 <= i <= 12 else 20 + i % 3,
                "cooked_kg": 200 + i,
                "plate_size_cm": 20 + i % 3,
                "meal": "lunch",
                "hostel": "Boys Hostel 1",
                "waste_kg": 18,
            }
        )
    finding = check_stuck_sensor(pd.DataFrame(rows), cfg)
    assert finding["present"] is True
    assert finding["details"]["column"] == "temp_c"
    assert finding["details"]["first_date"] == "2026-04-06"
    assert finding["details"]["last_date"] == "2026-04-13"
