"""Select, register, and document the measured Track A production candidate."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.data import PROJECT_ROOT
from src.train import COMPARISON_PATH, configure_tracking


MODEL_NAME = "telco-churn-classifier"
PRODUCTION_ALIAS = "production"
CANDIDATE_ALIAS = "candidate"
REGISTRY_EVIDENCE_PATH = PROJECT_ROOT / "reports" / "model-registry.json"
SELECTION_REPORT_PATH = PROJECT_ROOT / "reports" / "model-selection.md"
METRIC_COLUMNS = ("accuracy", "precision", "recall", "f1", "roc_auc")


@dataclass(frozen=True)
class SelectedModel:
    run_name: str
    run_id: str
    model_name: str
    metrics: dict[str, float]


def read_comparison(path: Path = COMPARISON_PATH) -> list[SelectedModel]:
    """Read the actual Phase 1 comparison output rather than hard-coding metrics."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No model rows found in {path}")
    return [
        SelectedModel(
            run_name=row["run_name"],
            run_id=row["run_id"],
            model_name=row["model"],
            metrics={column: float(row[column]) for column in METRIC_COLUMNS},
        )
        for row in rows
    ]


def select_model(models: list[SelectedModel]) -> SelectedModel:
    """Choose the model with the best minority-class F1/recall trade-off."""
    selected = max(models, key=lambda item: (item.metrics["f1"], item.metrics["recall"]))
    if selected.metrics["f1"] != max(item.metrics["f1"] for item in models):
        raise ValueError("Selected model must have the highest measured F1.")
    if selected.metrics["recall"] != max(item.metrics["recall"] for item in models):
        raise ValueError("Selected model must have the highest measured recall.")
    return selected


def selection_markdown(models: list[SelectedModel], selected: SelectedModel) -> str:
    rows = "\n".join(
        f"| {item.run_name} | {item.metrics['accuracy']:.6f} | {item.metrics['precision']:.6f} | "
        f"{item.metrics['recall']:.6f} | {item.metrics['f1']:.6f} | {item.metrics['roc_auc']:.6f} |"
        for item in models
    )
    logistic = next(item for item in models if item.run_name == "logistic-regression")
    return f"""# Model selection — Phase 2

Selection is based on the real values in `reports/mlflow-run-comparison.csv`.

| Run | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
{rows}

## Selected model

**{selected.run_name}** from MLflow run `{selected.run_id}` is registered as
`{MODEL_NAME}`. This imbalanced churn dataset requires more than accuracy:
the selected model has the highest recall ({selected.metrics['recall']:.6f})
and highest F1 ({selected.metrics['f1']:.6f}). Compared with logistic
regression, it identifies {selected.metrics['recall'] - logistic.metrics['recall']:.6f}
more of the positive class and improves F1 by
{selected.metrics['f1'] - logistic.metrics['f1']:.6f}, while giving up
{logistic.metrics['accuracy'] - selected.metrics['accuracy']:.6f} accuracy,
{logistic.metrics['precision'] - selected.metrics['precision']:.6f} precision,
and {logistic.metrics['roc_auc'] - selected.metrics['roc_auc']:.6f} ROC-AUC.

The decision therefore prioritizes correctly identifying churners and the
precision/recall balance, rather than the majority-class-favored accuracy.
"""


def _logged_model_for_run(client: MlflowClient, experiment_id: str, selected: SelectedModel):
    models = client.search_logged_models([experiment_id])
    matches = [model for model in models if model.source_run_id == selected.run_id and model.status.value == "READY"]
    if not matches:
        raise RuntimeError(f"No ready logged model found for selected run {selected.run_id}.")
    return max(matches, key=lambda model: model.creation_timestamp)


def _find_existing_version(client: MlflowClient, selected: SelectedModel, model_id: str) -> ModelVersion | None:
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    return next(
        (version for version in versions if version.run_id == selected.run_id and version.model_id == model_id),
        None,
    )


def _model_version_dict(version: ModelVersion) -> dict[str, Any]:
    return {
        "name": version.name,
        "version": version.version,
        "source": version.source,
        "run_id": version.run_id,
        "current_stage": version.current_stage,
        "creation_timestamp": version.creation_timestamp,
        "last_updated_timestamp": version.last_updated_timestamp,
        "tags": dict(version.tags),
    }


def register_selected_model() -> dict[str, Any]:
    """Register the selected run artifact and verify its lifecycle state from MLflow."""
    models = read_comparison()
    selected = select_model(models)
    tracking_uri, experiment_id = configure_tracking()
    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        client.get_registered_model(MODEL_NAME)
    except MlflowException:
        client.create_registered_model(MODEL_NAME, description="Track A Telco churn classifier.")

    logged_model = _logged_model_for_run(client, experiment_id, selected)
    version = _find_existing_version(client, selected, logged_model.model_id)
    if version is None:
        version = client.create_model_version(
            name=MODEL_NAME,
            source=logged_model.model_uri,
            run_id=selected.run_id,
            description="Selected from measured Phase 1 comparison.",
            model_id=logged_model.model_id,
        )
    version = client.get_model_version(MODEL_NAME, version.version)
    client.set_model_version_tag(MODEL_NAME, version.version, "selection_run_name", selected.run_name)
    client.set_model_version_tag(MODEL_NAME, version.version, "selection_reason", "highest_f1_and_recall_on_held_out_test")

    staging = client.transition_model_version_stage(MODEL_NAME, version.version, "Staging")
    client.set_registered_model_alias(MODEL_NAME, CANDIDATE_ALIAS, version.version)
    production = client.transition_model_version_stage(
        MODEL_NAME, version.version, "Production", archive_existing_versions=True
    )
    client.set_registered_model_alias(MODEL_NAME, PRODUCTION_ALIAS, version.version)

    verified_version = client.get_model_version(MODEL_NAME, version.version)
    verified_model = client.get_registered_model(MODEL_NAME)
    verified_alias = client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
    if verified_version.run_id != selected.run_id or verified_alias.version != version.version:
        raise RuntimeError("MLflow registry verification did not match the selected source run/version.")
    if verified_version.current_stage != "Production":
        raise RuntimeError("MLflow registry verification did not reach Production.")

    evidence = {
        "mlflow_version": mlflow.__version__,
        "registry_mechanism": "Classic stages remain supported by MLflow 3.16.1; aliases are also recorded for current serving.",
        "selected_model": {"run_name": selected.run_name, "model_name": selected.model_name, "metrics": selected.metrics},
        "registered_model": _model_version_dict(verified_version),
        "lifecycle_transitions": [
            {"observed_stage": staging.current_stage, "last_updated_timestamp": staging.last_updated_timestamp},
            {"observed_stage": production.current_stage, "last_updated_timestamp": production.last_updated_timestamp},
        ],
        "aliases": dict(verified_model.aliases),
        "verified_production_alias": _model_version_dict(verified_alias),
    }
    REGISTRY_EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    SELECTION_REPORT_PATH.write_text(selection_markdown(models, selected), encoding="utf-8")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Register the selected Track A model.")
    parser.add_argument("--register", action="store_true", help="Register and transition the selected MLflow run.")
    args = parser.parse_args()
    if not args.register:
        parser.error("Use --register to register the measured selected model.")
    evidence = register_selected_model()
    registered = evidence["registered_model"]
    print(f"Registered {registered['name']} version {registered['version']} from run {registered['run_id']}")
    print(f"Lifecycle: {[entry['observed_stage'] for entry in evidence['lifecycle_transitions']]}")
    print(f"Aliases: {evidence['aliases']}")


if __name__ == "__main__":
    main()
