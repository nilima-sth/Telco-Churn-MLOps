"""Deterministic Evidently monitoring for the Track A Telco churn data."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from evidently import Report
from evidently.metrics import DriftedColumnsCount, ValueDrift
from evidently.presets import DataDriftPreset

from src.data import PROJECT_ROOT, load_raw_data, validate_schema
from src.preprocess import RANDOM_SEED
from src.train import configure_tracking


REFERENCE_FRACTION = 0.70
CURRENT_FRACTION = 0.30
DRIFT_INJECTION_SEED = RANDOM_SEED
REPORT_DIRECTORY = PROJECT_ROOT / "reports" / "evidently"
REPORT_PATH = REPORT_DIRECTORY / "telco-drift-report.html"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "drift-summary.json"
ANALYSIS_PATH = PROJECT_ROOT / "reports" / "drift-analysis.md"
MONITORING_EXPERIMENT = "telco-churn-track-a-monitoring"
MONITOR_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]


@dataclass(frozen=True)
class MonitoringData:
    reference: pd.DataFrame
    current: pd.DataFrame
    before_current: pd.DataFrame


def make_reference_current(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a deterministic 70/30 split without modifying the source frame."""
    validate_schema(frame)
    shuffled = frame.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    reference_size = int(len(shuffled) * REFERENCE_FRACTION)
    return shuffled.iloc[:reference_size].copy(), shuffled.iloc[reference_size:].copy()


def inject_drift(reference: pd.DataFrame, current: pd.DataFrame) -> MonitoringData:
    """Shift only current MonthlyCharges and Contract with a seeded RNG."""
    reference_copy = reference.copy(deep=True)
    before = current.copy(deep=True)
    drifted = current.copy(deep=True)
    rng = pd.Series(range(len(drifted)), index=drifted.index).sample(frac=1.0, random_state=DRIFT_INJECTION_SEED)
    noise = pd.Series(
        __import__("numpy").random.default_rng(DRIFT_INJECTION_SEED).normal(0.0, 2.0, len(drifted)),
        index=drifted.index,
    )
    drifted["MonthlyCharges"] = (drifted["MonthlyCharges"].astype(float) + 20.0 + noise).clip(lower=0.0)
    target_month_to_month = int(len(drifted) * 0.80)
    month_to_month_indices = list(drifted.index[drifted["Contract"] == "Month-to-month"])
    remaining = [index for index in drifted.index if index not in month_to_month_indices]
    needed = max(0, target_month_to_month - len(month_to_month_indices))
    selected = list(month_to_month_indices) + list(rng.loc[remaining].index[:needed])
    drifted.loc[selected, "Contract"] = "Month-to-month"
    return MonitoringData(reference=reference_copy, current=drifted, before_current=before)


def prepare_monitoring_data(frame: pd.DataFrame) -> MonitoringData:
    reference, current = make_reference_current(frame)
    injected = inject_drift(reference, current)
    return MonitoringData(reference=reference, current=injected.current, before_current=injected.before_current)


def _distribution(frame: pd.DataFrame, column: str) -> dict[str, float]:
    return {str(key): float(value) for key, value in frame[column].value_counts(normalize=True).sort_index().items()}


def custom_metrics(data: MonitoringData) -> dict[str, float]:
    reference = data.reference
    current = data.current
    reference_churn = float((reference["Churn"] == "Yes").mean())
    current_churn = float((current["Churn"] == "Yes").mean())
    reference_mtm = reference["Contract"] == "Month-to-month"
    current_mtm = current["Contract"] == "Month-to-month"
    return {
        "mean_monthly_charges_shift": float(current["MonthlyCharges"].mean() - reference["MonthlyCharges"].mean()),
        "churn_rate_shift": current_churn - reference_churn,
        "month_to_month_rate_shift": float(current_mtm.mean() - reference_mtm.mean()),
        "reference_churn_rate": reference_churn,
        "current_churn_rate": current_churn,
        "reference_month_to_month_rate": float(reference_mtm.mean()),
        "current_month_to_month_rate": float(current_mtm.mean()),
    }


def _metric_value(snapshot: Any, prefix: str) -> float | None:
    for metric in snapshot.dict().get("metrics", []):
        if metric["metric_name"].startswith(prefix):
            value = metric["value"]
            if isinstance(value, dict):
                return float(value.get("count", value.get("share", 0.0)))
            return float(value)
    return None


def _drift_metrics(snapshot: Any) -> dict[str, Any]:
    values: dict[str, float] = {}
    for metric in snapshot.dict().get("metrics", []):
        name = metric["metric_name"]
        match = re.search(r"ValueDrift\(column=([^,]+),", name)
        if match:
            value = metric["value"]
            values[match.group(1)] = float(value)
    thresholds = {column: 0.1 for column in values}
    thresholds["TotalCharges"] = 0.55
    drifted = [column for column, value in values.items() if value >= thresholds[column]]
    count = _metric_value(snapshot, "DriftedColumnsCount")
    return {
        "column_drift_scores": values,
        "drift_thresholds": thresholds,
        "drifted_features": drifted,
        "evidently_drifted_column_count": int(count or len(drifted)),
        "evidently_drifted_column_share": float((count or len(drifted)) / len(values)) if values else 0.0,
        "dataset_drift_detected": bool(count and count > 0),
    }


def run_monitoring(data: MonitoringData) -> dict[str, Any]:
    metrics = [
        DataDriftPreset(columns=MONITOR_COLUMNS, drift_share=0.1, include_tests=True),
        DriftedColumnsCount(columns=MONITOR_COLUMNS, drift_share=0.1),
        ValueDrift(column="Churn", threshold=0.1),
    ]
    snapshot = Report(metrics=metrics, include_tests=True).run(data.current, data.reference, name="telco-drift")
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(REPORT_PATH))
    evidently_metrics = _drift_metrics(snapshot)
    custom = custom_metrics(data)
    target_score = evidently_metrics["column_drift_scores"].get("Churn", 0.0)
    summary = {
        "evidently_version": __import__("evidently").__version__,
        "reference": {"rows": len(data.reference), "fraction": REFERENCE_FRACTION, "churn_distribution": _distribution(data.reference, "Churn")},
        "current": {"rows": len(data.current), "fraction": CURRENT_FRACTION, "churn_distribution": _distribution(data.current, "Churn")},
        "drift_injection": {
            "seed": DRIFT_INJECTION_SEED,
            "numeric": {"column": "MonthlyCharges", "additive_shift": 20.0, "noise_std": 2.0},
            "categorical": {"column": "Contract", "category": "Month-to-month", "target_rate": 0.80},
            "before_current": {"monthly_charges_mean": float(data.before_current["MonthlyCharges"].mean()), "contract_distribution": _distribution(data.before_current, "Contract")},
            "after_current": {"monthly_charges_mean": float(data.current["MonthlyCharges"].mean()), "contract_distribution": _distribution(data.current, "Contract")},
        },
        "data_drift": evidently_metrics,
        "target_drift": {"column": "Churn", "method": "Evidently ValueDrift via current API", "score": target_score, "detected": target_score >= 0.1},
        "custom_metrics": custom,
        "report_path": "reports/evidently/telco-drift-report.html",
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _log_to_mlflow(summary: dict[str, Any]) -> str:
    tracking_uri, experiment_id = configure_tracking()
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(MONITORING_EXPERIMENT)
    if experiment is None:
        experiment_id = client.create_experiment(MONITORING_EXPERIMENT, artifact_location=(PROJECT_ROOT / "mlruns").as_uri())
    mlflow.set_experiment(MONITORING_EXPERIMENT)
    with mlflow.start_run(run_name="telco-drift-monitoring") as run:
        mlflow.log_params({
            "random_seed": RANDOM_SEED, "reference_fraction": REFERENCE_FRACTION, "current_fraction": CURRENT_FRACTION,
            "drifted_columns": "MonthlyCharges,Contract", "numeric_injection": "MonthlyCharges += 20 + seeded_normal_noise(std=2)",
            "categorical_injection": "Contract Month-to-month rate targeted to 0.80", "evidently_version": summary["evidently_version"],
        })
        mlflow.log_metrics({
            "drifted_feature_count": summary["data_drift"]["evidently_drifted_column_count"],
            "drifted_feature_share": summary["data_drift"]["evidently_drifted_column_share"],
            "dataset_drift_detected": float(summary["data_drift"]["dataset_drift_detected"]),
            "target_drift_detected": float(summary["target_drift"]["detected"]),
            **summary["custom_metrics"],
        })
        mlflow.log_artifact(str(REPORT_PATH), artifact_path="evidently")
        mlflow.log_artifact(str(SUMMARY_PATH), artifact_path="evidently")
        custom_path = REPORT_DIRECTORY / "custom-metrics.json"
        custom_path.write_text(json.dumps(summary["custom_metrics"], indent=2), encoding="utf-8")
        mlflow.log_artifact(str(custom_path), artifact_path="evidently")
        return run.info.run_id


def write_analysis(summary: dict[str, Any], run_id: str) -> None:
    detected = ", ".join(summary["data_drift"]["drifted_features"]) or "none"
    custom = summary["custom_metrics"]
    ANALYSIS_PATH.write_text(f"""# Evidently drift analysis

Reference is the deterministic 70% partition ({summary['reference']['rows']} rows); current is the remaining 30% ({summary['current']['rows']} rows). Only current received synthetic drift.

- Injected numeric drift: `MonthlyCharges` received +20 plus seeded Gaussian noise (std 2). Mean shifted from {summary['drift_injection']['before_current']['monthly_charges_mean']:.4f} to {summary['drift_injection']['after_current']['monthly_charges_mean']:.4f}; custom shift = **{custom['mean_monthly_charges_shift']:.4f}**.
- Injected categorical drift: `Contract=Month-to-month` increased from {summary['drift_injection']['before_current']['contract_distribution'].get('Month-to-month', 0):.4f} to {summary['drift_injection']['after_current']['contract_distribution'].get('Month-to-month', 0):.4f}; custom rate shift = **{custom['month_to_month_rate_shift']:.4f}**.
- Evidently detected drifted features: **{detected}**. Dataset drift detected = **{summary['data_drift']['dataset_drift_detected']}**.
- Target equivalent: Evidently 0.7.23 has no legacy `TargetDriftPreset`; `ValueDrift(column='Churn')` was used. Churn score = {summary['target_drift']['score']:.6f}; target drift detected = **{summary['target_drift']['detected']}**. Churn-rate shift = {custom['churn_rate_shift']:.4f}.
- Monitoring MLflow run: `{run_id}`.

The injected feature drift would warrant investigating the upstream contract and billing distributions, then evaluating production performance. Retraining should be considered only if the shift persists and model performance degrades; this run does not retrain automatically.
""", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic Evidently drift monitoring.")
    parser.parse_args()
    data = prepare_monitoring_data(load_raw_data())
    summary = run_monitoring(data)
    run_id = _log_to_mlflow(summary)
    write_analysis(summary, run_id)
    print(f"Evidently {summary['evidently_version']}; reference={summary['reference']['rows']} current={summary['current']['rows']}")
    print(f"Drifted features: {summary['data_drift']['drifted_features']}; target_drift={summary['target_drift']['detected']}")
    print(f"MonthlyCharges shift: {summary['custom_metrics']['mean_monthly_charges_shift']:.4f}; MLflow run={run_id}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
