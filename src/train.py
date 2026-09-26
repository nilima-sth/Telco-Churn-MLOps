"""Reproducible MLflow training experiments for Telco Customer Churn."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data import PROJECT_ROOT, load_raw_data
from src.evaluate import EvaluationResult, evaluate_binary_classifier, save_evaluation_figures
from src.preprocess import RANDOM_SEED, make_preprocessor, split_features_target


EXPERIMENT_NAME = "telco-churn-track-a"
TEST_SIZE = 0.20
FIGURE_DIRECTORY = PROJECT_ROOT / "reports" / "figures"
COMPARISON_PATH = PROJECT_ROOT / "reports" / "mlflow-run-comparison.csv"


@dataclass(frozen=True)
class ModelConfig:
    run_name: str
    model_name: str
    hyperparameters: dict[str, Any]

    def build_estimator(self):
        if self.model_name == "logistic_regression":
            return LogisticRegression(**self.hyperparameters)
        if self.model_name == "random_forest":
            return RandomForestClassifier(**self.hyperparameters)
        raise ValueError(f"Unsupported model configuration: {self.model_name}")


MODEL_CONFIGS = (
    ModelConfig(
        run_name="logistic-regression",
        model_name="logistic_regression",
        hyperparameters={"C": 1.0, "max_iter": 1000, "solver": "lbfgs", "random_state": RANDOM_SEED},
    ),
    ModelConfig(
        run_name="random-forest-baseline",
        model_name="random_forest",
        hyperparameters={"n_estimators": 100, "max_depth": 8, "min_samples_split": 2, "min_samples_leaf": 1, "random_state": RANDOM_SEED, "n_jobs": -1},
    ),
    ModelConfig(
        run_name="random-forest-tuned",
        model_name="random_forest",
        hyperparameters={"n_estimators": 300, "max_depth": 16, "min_samples_split": 10, "min_samples_leaf": 4, "class_weight": "balanced", "random_state": RANDOM_SEED, "n_jobs": -1},
    ),
)


@dataclass(frozen=True)
class SplitData:
    x_train: Any
    x_test: Any
    y_train: Any
    y_test: Any


@dataclass(frozen=True)
class RunResult:
    run_name: str
    run_id: str
    model_name: str
    hyperparameters: dict[str, Any]
    metrics: EvaluationResult
    confusion_matrix_path: str
    roc_curve_path: str


def create_split() -> SplitData:
    """Create the sole stratified held-out test split used by every experiment."""
    features, target = split_features_target(load_raw_data())
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=target
    )
    return SplitData(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)


def make_model_pipeline(config: ModelConfig, training_features) -> Pipeline:
    return Pipeline([
        ("preprocessor", make_preprocessor(training_features)),
        ("estimator", config.build_estimator()),
    ])


def configure_tracking() -> tuple[str, str]:
    tracking_uri = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').resolve()}"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = client.create_experiment(EXPERIMENT_NAME, artifact_location=(PROJECT_ROOT / "mlruns").as_uri())
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(EXPERIMENT_NAME)
    return tracking_uri, experiment_id


def run_experiment(config: ModelConfig, split: SplitData) -> RunResult:
    """Fit, evaluate, figure, and log one complete preprocessing-plus-model pipeline."""
    pipeline = make_model_pipeline(config, split.x_train)
    with mlflow.start_run(run_name=config.run_name) as active_run:
        pipeline.fit(split.x_train, split.y_train)
        predictions = pipeline.predict(split.x_test)
        probabilities = pipeline.predict_proba(split.x_test)[:, 1]
        metrics = evaluate_binary_classifier(split.y_test, predictions, probabilities)
        confusion_path, roc_path = save_evaluation_figures(
            split.y_test, predictions, probabilities, run_name=config.run_name, output_directory=FIGURE_DIRECTORY
        )
        parameters = {
            "model_type": config.model_name,
            "random_seed": RANDOM_SEED,
            "test_size": TEST_SIZE,
            "train_rows": len(split.x_train),
            "test_rows": len(split.x_test),
            "positive_class": "Churn=Yes=1",
            "total_charges_strategy": "blank_to_nan_then_median_imputation",
            "categorical_encoding": "one_hot_handle_unknown_ignore",
            **config.hyperparameters,
        }
        mlflow.log_params({key: str(value) for key, value in parameters.items()})
        mlflow.log_metrics(metrics.metrics())
        mlflow.log_artifact(str(confusion_path), artifact_path="figures")
        mlflow.log_artifact(str(roc_path), artifact_path="figures")
        with tempfile.TemporaryDirectory(prefix="tracka_run_") as temporary_directory:
            artifact_directory = Path(temporary_directory)
            (artifact_directory / "metrics.json").write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
            (artifact_directory / "run_summary.json").write_text(json.dumps({"run_name": config.run_name, "parameters": parameters}, indent=2), encoding="utf-8")
            mlflow.log_artifacts(str(artifact_directory))
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        run_id = active_run.info.run_id
    return RunResult(
        run_name=config.run_name, run_id=run_id, model_name=config.model_name,
        hyperparameters=config.hyperparameters, metrics=metrics,
        confusion_matrix_path=str(confusion_path), roc_curve_path=str(roc_path),
    )


def write_comparison(results: list[RunResult], path: Path = COMPARISON_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_name", "run_id", "model", "accuracy", "precision", "recall", "f1", "roc_auc", "important_hyperparameters"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "run_name": result.run_name, "run_id": result.run_id, "model": result.model_name,
                **result.metrics.metrics(), "important_hyperparameters": json.dumps(result.hyperparameters, sort_keys=True),
            })


def format_comparison(results: list[RunResult]) -> str:
    header = "run_name                     accuracy  precision  recall  f1      roc_auc"
    rows = [header]
    for result in results:
        metric = result.metrics
        rows.append(f"{result.run_name:<28} {metric.accuracy:>8.3f}  {metric.precision:>9.3f}  {metric.recall:>6.3f}  {metric.f1:>6.3f}  {metric.roc_auc:>7.3f}")
    return "\n".join(rows)


def run_all() -> tuple[list[RunResult], SplitData, str, str]:
    tracking_uri, experiment_id = configure_tracking()
    split = create_split()
    results = [run_experiment(config, split) for config in MODEL_CONFIGS]
    write_comparison(results)
    return results, split, tracking_uri, experiment_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible MLflow Telco Churn experiments.")
    parser.add_argument("--all", action="store_true", help="Run all configured baseline experiments.")
    args = parser.parse_args()
    if not args.all:
        parser.error("Use --all to run the three configured experiments.")
    results, split, tracking_uri, experiment_id = run_all()
    print(f"MLflow experiment: {EXPERIMENT_NAME} ({experiment_id})")
    print(f"Tracking URI: {tracking_uri}")
    print(f"Split: train={len(split.x_train)} test={len(split.x_test)}; train churn={int(split.y_train.sum())}, test churn={int(split.y_test.sum())}")
    print(format_comparison(results))
    print(f"Comparison CSV: {COMPARISON_PATH}")
    for result in results:
        print(f"{result.run_name}: run_id={result.run_id}; confusion={result.confusion_matrix_path}; roc={result.roc_curve_path}")


if __name__ == "__main__":
    main()
