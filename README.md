# SCSE3040 MLOps Mini Project --- Group G151

## Problem: MP301 --- Hostel Mess Food Waste (Dehradun)

**Client:** Dehradun University Hostel Office  
**Faculty:** Dr. Ravi Pratap | **Batch:** 63  

---

### Group Members

| Enrolment No. | Name | Role | Primary Responsibilities |
|---|---|---|---|
| **S24CSEU2317** | **Nakshatra Gupta** | **Data Lead** | DVC setup, 7 data checks audit, cleaning pipeline, `findings.yaml` data section |
| S24CSEU2230 | Gunjan Kandpal | Model Lead | Training, MLflow tracking and model registry, holdout prediction |
| S24CSEU1197 | Kanan Singhal | Service Lead | FastAPI application, Pydantic schemas, validation rules, Dockerfile |
| S24CSEU2006 | Aarna Verma | Ops Lead | CI/CD GitHub Actions, Kubernetes manifests, structured logging |
| S24CSEU2297 | Riya Bhati | Member / Contributor | Drift monitoring (PSI), retraining evaluation, holdout validation |

---

### Project Overview

The hostel mess of Dehradun University serves hundreds of students daily across three hostels:
* Boys Hostel 1
* Boys Hostel 2
* Girls Hostel

The objective is to accurately predict post-meal food waste (`waste_kg`) to optimize meal preparation quantities, cut food wastage, and reduce operational costs.

---

### Repository Layout

```
README.md                           # Group and project overview, team member details
requirements.txt                    # Pinned library dependencies
params.yaml                         # Central configuration parameters
src/mp301/                          # Source code package
  ├── __init__.py
  ├── data_checks.py                # 7 data problems inspection suite
  ├── cleaning.py                   # Data cleaning pipeline
  └── utils.py                      # YAML config loader and hashing utilities
tests/                              # Automated test suite
  ├── test_data_cleaning.py         # Unit tests for data cleaning logic
data/
  ├── raw/
  │   ├── train.csv.dvc             # DVC tracking for raw training data
  │   ├── later.csv.dvc             # DVC tracking for monitoring data
  │   └── holdout_features.csv      # Unlabeled holdout features
  └── processed/                    # Cleaned and transformed datasets
reports/
  └── findings.yaml                 # Formal project findings and data audit report
.dvc/                               # DVC configuration
```

---

### Data Lead Deliverables (Stages 1 & 2)

#### 1. DVC Tracking
Raw data files `train.csv` and `later.csv` are tracked under DVC to ensure version control and data reproducibility. Hashes are locked in `.dvc` tracking files, and CSV files are excluded from Git tracking via `.gitignore`.

#### 2. The Seven Data Problems Audit
Audited `train.csv` against all 7 data anomalies described in the Student Handbook:
1. **Leaking Column:** `waste_weighed_at_bin` was identified as a post-event measurement, absent from holdout data and strongly correlated with the target. Marked `present: true`.
2. **Unit Mixup:** `plate_size_cm` in `Boys Hostel 1` was recorded in millimeters (mm) instead of centimeters (cm). Conversion factor: `10`. Marked `present: true`.
3. **Sentinel Value:** `temp_c` contained `-999.0` for missing readings in 58 rows. Marked `present: true`.
4. **Duplicate Rows:** Checked via `df.duplicated().sum()`. Exactly 0 extra rows found. Marked `present: false`.
5. **Stuck Sensor:** Verified consecutive readings across dates. No stuck sensor runs detected. Marked `present: false`.
6. **Target Typos:** Exactly 12 rows had `waste_kg` typed 10 times too big (values > 200 kg). Marked `present: true`.
7. **Spelling Variants:** Inspected unique categories in `meal` and `hostel`. All match clean categories in `later.csv`. Marked `present: false`.

#### 3. How to Run the Data Checks Live (Viva Verification)
```bash
python -m src.mp301.data_checks
```

#### 4. How to Run Data Cleaning
```bash
python -m src.mp301.cleaning
```

#### 5. How to Run Unit Tests
```bash
pytest tests/
```

---

## Stage 2 — Training, MLflow and service (added)

The stage-1 files above are untouched. The stage-2 additions live alongside them:

```
src/mp301/config_ml.py      # reads the existing params.yaml and adapts it for stage 2
src/mp301/checks_ml.py      # seven checks with detailed evidence (viva demo)
src/mp301/cleaning_ml.py    # deterministic cleaning driven by those checks
src/mp301/model.py          # sklearn preprocessing + regressor factory
src/mp301/train.py          # 3 experiments + v1 champion registration
app/main.py                 # FastAPI /predict and /health
tests/test_stage2_ml.py     # 6 data/cleaning tests
tests/test_api.py           # 5 TestClient tests, including both 422 rules
mlflow.db                   # committed tracking store (4 runs)
models/model.joblib         # v1 champion used by the API
reports/stage2_summary.json # exact run ids and metrics
reports/stage2_details.yaml # detailed evidence for every data finding
Dockerfile, .dockerignore, .github/workflows/ci.yml, k8s/
```

### Already-generated results

- `data_sha256`: `3485c8e6f60956e7a6f46807cd821f1d79bbb89e47749dfd10a6d8147ffaa01c`
- 3 candidate runs + 1 champion run in `mlflow.db`
- Best v1 = `ridge`, validation MAE `2.2523`, later MAE `8.9024`
- Registered model `mp301_waste_model`, version **1**, alias **`champion`**
- Champion run id: `cd10f7758bf94aa79675d648cf50550b`
- `pytest -q` → **21 passed**

**Do not rerun `python -m src.mp301.train --fresh`** unless you intentionally want new run ids; the database and `findings.yaml` are already filled.

To inspect the service locally:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
