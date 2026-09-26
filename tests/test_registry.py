from src.registry import SelectedModel, select_model, selection_markdown


def test_selection_prioritizes_highest_f1_and_recall() -> None:
    models = [
        SelectedModel("logistic-regression", "v1", "logistic_regression", {"accuracy": 0.806, "precision": 0.657, "recall": 0.559, "f1": 0.604, "roc_auc": 0.842}),
        SelectedModel("random-forest-tuned", "v3", "random_forest", {"accuracy": 0.757, "precision": 0.530, "recall": 0.767, "f1": 0.627, "roc_auc": 0.841}),
    ]
    selected = select_model(models)

    assert selected.run_id == "v3"
    report = selection_markdown(models, selected)
    assert "highest recall (0.767000)" in report
    assert "highest F1 (0.627000)" in report
