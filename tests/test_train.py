import csv

from src.evaluate import EvaluationResult
from src.train import MODEL_CONFIGS, RunResult, create_split, make_model_pipeline, write_comparison


def test_model_configurations_are_materially_different() -> None:
    assert [config.model_name for config in MODEL_CONFIGS] == ["logistic_regression", "random_forest", "random_forest"]
    assert MODEL_CONFIGS[1].hyperparameters["n_estimators"] != MODEL_CONFIGS[2].hyperparameters["n_estimators"]
    assert MODEL_CONFIGS[1].hyperparameters["max_depth"] != MODEL_CONFIGS[2].hyperparameters["max_depth"]


def test_pipeline_fits_and_exposes_positive_class_probabilities() -> None:
    split = create_split()
    pipeline = make_model_pipeline(MODEL_CONFIGS[0], split.x_train)
    pipeline.fit(split.x_train, split.y_train)

    probabilities = pipeline.predict_proba(split.x_test.iloc[:3])
    assert probabilities.shape == (3, 2)
    assert pipeline.predict(split.x_test.iloc[:3]).shape == (3,)


def test_comparison_output_contains_real_run_fields(tmp_path) -> None:
    result = RunResult(
        run_name="example", run_id="run-123", model_name="logistic_regression", hyperparameters={"C": 1.0},
        metrics=EvaluationResult(accuracy=0.8, precision=0.7, recall=0.6, f1=0.64, roc_auc=0.75, confusion_matrix=[[1, 2], [3, 4]]),
        confusion_matrix_path="cm.png", roc_curve_path="roc.png",
    )
    path = tmp_path / "comparison.csv"
    write_comparison([result], path)

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["run_id"] == "run-123"
    assert rows[0]["roc_auc"] == "0.75"
