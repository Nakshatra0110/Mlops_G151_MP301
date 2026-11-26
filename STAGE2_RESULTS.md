# Stage 2 results (computed on your files)

## Data fingerprint

`data_sha256 = 3485c8e6f60956e7a6f46807cd821f1d79bbb89e47749dfd10a6d8147ffaa01c`

This was logged in **all four** MLflow runs and written to `reports/findings.yaml`.

## The seven findings

| Check | Result | Evidence |
|---|---|---|
| `leaking_column` | **present** | `waste_weighed_at_bin`, absent from holdout; corr 0.998686 with typo-corrected `waste_kg`, MAE 0.4793 kg |
| `unit_mixup` | **present** | `plate_size_cm`, `Boys Hostel 1`, factor **10**, divide by 10 (median 269.0 → 26.9 vs peer 27.4) |
| `sentinel_value` | **present** | `temp_c = -999`, **58** cells |
| `duplicate_rows` | absent | 0 extra copies |
| `stuck_sensor` | absent | no run of ≥7 rows over ≥5 days |
| `target_typos` | **present** | 12 values 236–587 kg, factor **10**, gap 68.2 → 236 (3.46x) |
| `spelling_variants` | absent | meal and hostel levels already clean in train and later |

## MLflow

Tracking URI: `sqlite:///mlflow.db` (committed). Registered model: `mp301_waste_model`, version **1**, alias **champion**.

| Run | Model | `mae_val` | `mae_later` |
|---|---|---:|---:|
| `8ad69f6b351a429bb49712daa6677575` | ridge | **2.2523** | 9.0089 |
| `30f2537679b243fda7fdb359f83fa787` | random_forest | 2.8494 | 13.9934 |
| `de02f08421d24bd5983bc0a1c57f8e0c` | hist_gb | 2.3690 | 13.1824 |
| `cd10f7758bf94aa79675d648cf50550b` | **v1 champion (ridge)** | — | **8.9024** |

The champion run id is also stored at the top level of `reports/findings.yaml` as `run_id` / `champion_run_id`.

## Tests

`pytest -q` → **21 passed** in the merged repository (10 existing stage-1 tests + 6 stage-2 data tests + 5 TestClient tests).

## Important note about `findings.yaml`

The repository's original `findings.yaml` template was preserved. Its `data_issues` and `tracking` sections are now filled; `monitoring` is intentionally left null until stage 4. Detailed evidence lives in `reports/stage2_details.yaml`.
