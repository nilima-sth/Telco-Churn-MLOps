# Model selection — Phase 2

Selection is based on the real values in `reports/mlflow-run-comparison.csv`.

| Run | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| logistic-regression | 0.805536 | 0.657233 | 0.558824 | 0.604046 | 0.841861 |
| random-forest-baseline | 0.804116 | 0.675000 | 0.505348 | 0.577982 | 0.841496 |
| random-forest-tuned | 0.757275 | 0.529520 | 0.767380 | 0.626638 | 0.841249 |

## Selected model

**random-forest-tuned** from MLflow run `febd9e029d3a4738b07e2c3342c51551` is registered as
`telco-churn-classifier`. This imbalanced churn dataset requires more than accuracy:
the selected model has the highest recall (0.767380)
and highest F1 (0.626638). Compared with logistic
regression, it identifies 0.208556
more of the positive class and improves F1 by
0.022591, while giving up
0.048261 accuracy,
0.127712 precision,
and 0.000612 ROC-AUC.

The decision therefore prioritizes correctly identifying churners and the
precision/recall balance, rather than the majority-class-favored accuracy.
