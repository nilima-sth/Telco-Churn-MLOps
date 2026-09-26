"""Shared binary-classification evaluation for Track A training runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: list[list[int]]

    def metrics(self) -> dict[str, float]:
        values = asdict(self)
        values.pop("confusion_matrix")
        return values


def evaluate_binary_classifier(y_true, predictions, probabilities) -> EvaluationResult:
    """Evaluate Churn=Yes=1 with predictions and positive-class probabilities."""
    return EvaluationResult(
        accuracy=float(accuracy_score(y_true, predictions)),
        precision=float(precision_score(y_true, predictions, pos_label=1, zero_division=0)),
        recall=float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        f1=float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        confusion_matrix=confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    )


def save_evaluation_figures(
    y_true,
    predictions,
    probabilities,
    *,
    run_name: str,
    output_directory: Path,
) -> tuple[Path, Path]:
    """Create deterministic, submission-friendly confusion and ROC figures."""
    output_directory.mkdir(parents=True, exist_ok=True)
    confusion_path = output_directory / f"{run_name}-confusion-matrix.png"
    roc_path = output_directory / f"{run_name}-roc-curve.png"

    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, predictions, labels=[0, 1], display_labels=["No churn", "Churn"], ax=axis, colorbar=False
    )
    axis.set_title(f"{run_name}: confusion matrix")
    figure.tight_layout()
    figure.savefig(confusion_path, dpi=150)
    plt.close(figure)

    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities, pos_label=1)
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(false_positive_rate, true_positive_rate, label="Model ROC")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title=f"{run_name}: ROC curve")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(roc_path, dpi=150)
    plt.close(figure)
    return confusion_path, roc_path
