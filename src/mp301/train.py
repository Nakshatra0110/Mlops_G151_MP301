"""Stage 2: experiment tracking, registry and the v1 champion.

Run from the repository root:
    python -m src.mp301.train --fresh
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from .checks_ml import run_all_checks
from .cleaning_ml import clean_train, normalise_categories, prepare_xy
from .config_ml import REPO_ROOT, find_csv, load_params, resolve
from .model import build_pipeline


def sha256_file(path: Path) -> str:
    """The exact fingerprint the handbook asks for in every MLflow run."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _deep_merge(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


class _NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data) -> bool:
        return True


def _template_updates(checks: dict, params: dict, summary: dict) -> dict:
    """Fill the exact keys of the findings.yaml template from the zip."""
    leak = checks["leaking_column"]
    unit = checks["unit_mixup"]
    sentinel = checks["sentinel_value"]
    dupes = checks["duplicate_rows"]
    stuck = checks["stuck_sensor"]
    typo = checks["target_typos"]
    spelling = checks["spelling_variants"]
    first_sentinel = sentinel["details"][0] if sentinel["details"] else {}

    data_issues = {
        "leaking_column": {
            "present": bool(leak["present"]),
            "column": leak["column"],
        },
        "unit_mixup": {
            "present": bool(unit["present"]),
            "column": unit["column"],
            "group_value": unit["group"],
            "factor": unit["factor"],
        },
        "sentinel_value": {
            "present": bool(sentinel["present"]),
            "column": first_sentinel.get("column"),
            "value": first_sentinel.get("value"),
            "count": first_sentinel.get("count"),
        },
        "duplicate_rows": {
            "present": bool(dupes["present"]),
            "count": dupes["extra_copies"] if dupes["present"] else None,
        },
        "stuck_sensor": {
            "present": bool(stuck["present"]),
            "column": stuck["details"].get("column"),
            "start_date": stuck["details"].get("first_date"),
            "end_date": stuck["details"].get("last_date"),
        },
        "target_typos": {
            "present": bool(typo["present"]),
            "count": typo["count"] if typo["present"] else None,
        },
        "spelling_variants": {
            "present": bool(spelling["present"]),
            "column": spelling["column"],
            "true_levels": spelling["true_levels"],
        },
    }
    return {
        "group_id": params["problem"]["group_id"],
        "problem_id": params["problem"]["id"],
        "data_issues": data_issues,
        "tracking": {
            "experiment_name": summary["experiment"],
            "best_run_id": summary["champion_run_id"],
            "registered_model_name": params["model"]["registered_name"],
            "champion_version": summary["champion_version"],
        },
    }


def _write_template_findings(checks: dict, params: dict, summary: dict) -> list[Path]:
    updates = _template_updates(checks, params, summary)
    written: list[Path] = []
    for relative in ["findings.yaml", params["paths"]["findings"]]:
        target = REPO_ROOT / relative
        existing: dict = {}
        if target.exists():
            with target.open("r", encoding="utf-8") as handle:
                existing = yaml.safe_load(handle) or {}
        merged = _deep_merge(existing, updates)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            yaml.dump(merged, handle, Dumper=_NoAliasDumper, sort_keys=False, allow_unicode=True)
        written.append(target)

    detail_path = REPO_ROOT / "reports" / "stage2_details.yaml"
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    with detail_path.open("w", encoding="utf-8") as handle:
        yaml.dump(
            {"data_sha256": summary["data_sha256"], "checks": checks, "summary": summary},
            handle,
            Dumper=_NoAliasDumper,
            sort_keys=False,
            allow_unicode=True,
        )
    return written


def _log_model_compat(model, artifact_name: str, signature=None):
    """MLflow 3 renamed artifact_path to name; support both APIs."""
    params = inspect.signature(mlflow.sklearn.log_model).parameters
    kwargs: dict = {}
    if "name" in params:
        kwargs["name"] = artifact_name
    elif "artifact_path" in params:
        kwargs["artifact_path"] = artifact_name
    else:
        raise RuntimeError("mlflow.sklearn.log_model has no name/artifact_path argument")
    if "serialization_format" in params:
        kwargs["serialization_format"] = "cloudpickle"
    if signature is not None and "signature" in params:
        kwargs["signature"] = signature
    return mlflow.sklearn.log_model(model, **kwargs)


def run_stage2(params_path: str | Path | None = None, fresh: bool = False) -> dict:
    params = load_params(params_path)
    train_path = find_csv("train.csv")
    later_path = find_csv("later.csv")
    try:
        holdout_path = find_csv("holdout_features.csv")
    except FileNotFoundError:
        holdout_path = None

    sha = sha256_file(train_path)
    train_raw = pd.read_csv(train_path)
    later_raw = pd.read_csv(later_path)
    holdout_raw = pd.read_csv(holdout_path) if holdout_path else None

    checks = run_all_checks(train_raw, later_raw, holdout_raw, params)
    train_clean = clean_train(train_raw, checks, params, later=later_raw)
    later_clean = normalise_categories(later_raw, checks, params, reference=later_raw)

    X, y = prepare_xy(train_clean, params)
    X_later, y_later = prepare_xy(later_clean, params)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X,
        y,
        test_size=float(params["model"]["test_size"]),
        random_state=int(params["model"]["random_state"]),
    )

    os.chdir(REPO_ROOT)
    db_path = resolve(params["paths"]["mlflow_db"])
    if fresh and db_path.exists():
        db_path.unlink()
    # Required by the handbook: SQLite in the repository root, not ./mlruns.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment_name = f"mp301-{params['problem']['group_id']}-v1"
    mlflow.set_experiment(experiment_name)

    results: list[dict] = []
    for candidate in params["model"]["candidates"]:
        with mlflow.start_run(run_name=f"v1-{candidate['name']}") as run:
            mlflow.log_params(
                {
                    "data_sha256": sha,
                    "model_kind": candidate["kind"],
                    "model_name": candidate["name"],
                    **{k: v for k, v in candidate.items() if k not in {"name", "kind"}},
                }
            )
            model = build_pipeline(candidate, params)
            model.fit(X_fit, y_fit)
            mae_val = float(mean_absolute_error(y_val, model.predict(X_val)))
            mae_later = float(mean_absolute_error(y_later, model.predict(X_later)))
            mlflow.log_metrics({"mae_val": mae_val, "mae_later": mae_later})
            results.append(
                {
                    "name": candidate["name"],
                    "run_id": run.info.run_id,
                    "mae_val": mae_val,
                    "mae_later": mae_later,
                }
            )

    best = min(results, key=lambda item: item["mae_val"])
    best_candidate = next(c for c in params["model"]["candidates"] if c["name"] == best["name"])

    with mlflow.start_run(run_name="v1-champion") as run:
        champion_run_id = run.info.run_id
        mlflow.log_params(
            {
                "data_sha256": sha,
                "model_kind": best_candidate["kind"],
                "model_name": best_candidate["name"],
                "selected_by": "mae_val",
                **{
                    k: v
                    for k, v in best_candidate.items()
                    if k not in {"name", "kind"}
                },
            }
        )
        model = build_pipeline(best_candidate, params)
        model.fit(X, y)
        mae_later_final = float(mean_absolute_error(y_later, model.predict(X_later)))
        mlflow.log_metrics(
            {
                "mae_later": mae_later_final,
                "mae_validation_best_candidate": best["mae_val"],
            }
        )
        try:
            signature = infer_signature(X.head(5), model.predict(X.head(5)))
        except Exception:
            signature = None
        info = _log_model_compat(model, "model", signature=signature)
        model_uri = getattr(info, "model_uri", None) or f"runs:/{champion_run_id}/model"
        registered_name = params["model"]["registered_name"]
        version = mlflow.register_model(model_uri=model_uri, name=registered_name)
        MlflowClient().set_registered_model_alias(
            registered_name, params["model"]["alias"], version.version
        )

        model_out = resolve(params["paths"]["model_out"])
        model_out.parent.mkdir(parents=True, exist_ok=True)
        import joblib

        joblib.dump(model, model_out)

    summary = {
        "experiment": experiment_name,
        "data_sha256": sha,
        "candidate_run_ids": [r["run_id"] for r in results],
        "candidates": results,
        "champion_run_id": champion_run_id,
        "champion_version": int(version.version),
        "best_candidate": best["name"],
        "best_mae_validation": best["mae_val"],
        "best_mae_later": mae_later_final,
        "rows_used": int(len(train_clean)),
        "raw_rows": int(len(train_raw)),
        "features": list(X.columns),
    }
    summary_path = REPO_ROOT / "reports" / "stage2_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    written = _write_template_findings(checks, params, summary)
    print(json.dumps(summary, indent=2))
    print("Updated: " + ", ".join(str(p.relative_to(REPO_ROOT)) for p in written))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="MP301 stage-2 training")
    parser.add_argument("--params", default=None, help="Path to params.yaml")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete mlflow.db first (use only for a clean first run)",
    )
    args = parser.parse_args()
    run_stage2(args.params, fresh=args.fresh)


if __name__ == "__main__":
    main()
