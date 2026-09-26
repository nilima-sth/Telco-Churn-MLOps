import numpy as np

from src.evaluate import evaluate_binary_classifier


def test_binary_metrics_use_churn_yes_as_positive_class() -> None:
    result = evaluate_binary_classifier(
        np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), np.array([0.1, 0.8, 0.7, 0.9])
    )

    assert result.confusion_matrix == [[1, 1], [0, 2]]
    assert result.accuracy == 0.75
    assert result.precision == 2 / 3
    assert result.recall == 1.0
    assert result.f1 == 0.8
    assert result.roc_auc == 0.75
