# Week 17 Track A — Telco Customer Churn MLOps

## Dataset

Raw data is the IBM Telco Customer Churn sample, saved as
`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

Source: https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv

The downloaded file has 7,043 records, 21 columns, no null values reported by
pandas, no duplicate rows, and 11 blank `TotalCharges` values (all associated
with zero tenure). The target distribution is 5,174 `No` and 1,869 `Yes`.

## Structure

```text
data/raw/                 # verified source CSV
data/processed/           # future generated data, ignored
src/data.py               # schema loading and profiling
src/preprocess.py         # train-safe feature/target preparation
tests/                    # Phase 0 schema and preprocessing tests
reports/figures/          # future generated figures, ignored
reports/evidently/        # future generated reports, ignored
models/                   # future generated models, ignored
```

## Setup and validation

Python 3.12 and uv are required.

```bash
uv sync
uv lock --check
uv run python -m compileall src tests
uv run pytest -q
```

## Phase 1 — reproducible MLflow experiments

Run all three experiments with:

```bash
uv run python -m src.train --all
```

The single reproducible split uses `random_state=42`, `test_size=0.20`, and
stratification by `Churn`: 5,634 training rows (1,495 churn-positive) and
1,409 held-out test rows (374 churn-positive). `TotalCharges` is converted from
blank strings to missing values and median-imputed inside the pipeline; numeric
features are scaled and categorical features are one-hot encoded. The fitted
preprocessor and estimator are logged together as one sklearn pipeline.

| Run | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | MLflow run ID |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| V1 | Logistic regression | 0.806 | 0.657 | 0.559 | 0.604 | 0.842 | `bf2be21537b04ba2ac2e0a6128e7595d` |
| V2 | Random forest baseline (`n_estimators=100`, `max_depth=8`) | 0.804 | 0.675 | 0.505 | 0.578 | 0.841 | `1417d492fc9b4e1686280436fa6038fe` |
| V3 | Tuned random forest (`n_estimators=300`, `max_depth=16`, `min_samples_split=10`, `min_samples_leaf=4`, `class_weight=balanced`) | 0.757 | 0.530 | 0.767 | 0.627 | 0.841 | `febd9e029d3a4738b07e2c3342c51551` |

The tuned forest has the strongest F1/recall trade-off; logistic regression has
the highest ROC-AUC. Phase 1 logs this evidence only—there is no registered or
served candidate model.

The comparison table is saved at `reports/mlflow-run-comparison.csv`. Each of
the three completed MLflow runs contains the five metrics, model/split
parameters, `figures/` with a confusion matrix and ROC curve, `metrics.json`,
`run_summary.json`, and `model/` containing the serialized full pipeline.

Inspect them locally with:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5056
```

Then open http://127.0.0.1:5056 and select the `telco-churn-track-a`
experiment. The UI was verified reachable after the Phase 1 runs completed.

## Phase 2 — registry lifecycle and serving

### Selection and registration

The selected model is V3, `random-forest-tuned`, from source run
`febd9e029d3a4738b07e2c3342c51551`. It has the strongest held-out churn
identification trade-off: recall **0.767380** and F1 **0.626638**, compared
with logistic regression's recall **0.558824** and F1 **0.604046**. This gains
0.208556 recall and 0.022591 F1, while sacrificing 0.048261 accuracy, 0.127712
precision, and 0.000612 ROC-AUC. Accuracy alone is not decisive for this
imbalanced dataset.

It is registered as `telco-churn-classifier` version **2** in MLflow 3.16.1.
The version's registered source is the MLflow 3 logged-model URI
`models:/m-f820900a1a044caca47e9795d1fddee7`, with traceability to the source
run above. Classic stages are deprecated in MLflow 3.16.1 but remain supported
by the installed registry API; the verified lifecycle was `Staging` then
`Production`. The current `candidate` and `production` aliases both point to
version 2, and serving loads `models:/telco-churn-classifier@production`.

Registry facts are read back from MLflow and recorded in
[`reports/model-registry.json`](reports/model-registry.json). The full metric
comparison and selection rationale are in
[`reports/model-selection.md`](reports/model-selection.md).

### Serve the registered production model

```bash
uv run uvicorn src.serve:app --host 127.0.0.1 --port 8000
```

`GET /health` returns `{"status":"ok"}`. `POST /predict` accepts the raw
19-feature Telco schema (no target, ID, or pre-encoding required), validates it
with Pydantic, and returns the prediction, churn label, real probability, and
registry model/version. For example:

```json
{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
  "tenure": 60, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
  "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes",
  "TechSupport": "Yes", "StreamingTV": "No", "StreamingMovies": "No",
  "Contract": "Two year", "PaperlessBilling": "No",
  "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 65.0, "TotalCharges": 3900.0
}
```

The live response for that request was `{"prediction": 0, "churn": "No",
"churn_probability": 0.04889424464424465, "model_name":
"telco-churn-classifier", "model_version": "2"}`. A contrasting higher-risk
request returned `Yes` with probability `0.9589888437027753`. Both raw requests
and real responses are captured in
[`reports/serving-smoke-test.json`](reports/serving-smoke-test.json).

## Phase 3 — Evidently drift monitoring

Monitoring uses Evidently **0.7.23** and its current `Report`/`DataDriftPreset`
API. The deterministic reference set is 70% of the shuffled raw dataset
(4,930 rows); current is the remaining 30% (2,113 rows). Both retain the
`Churn` target for target-distribution analysis. Only current is modified.

The runner is:

```bash
uv run python -m src.drift
```

For a reproducible synthetic monitoring test, current `MonthlyCharges` receives
`+20` plus seeded Gaussian noise (standard deviation 2), and
`Contract=Month-to-month` is increased from **0.5390** to **0.7998**. The
measured monthly-charge mean shifts from **64.4877** to **84.3798**.

Evidently detected exactly the intentionally changed features:
`Contract` and `MonthlyCharges`. Dataset drift was **True** (2 drifted columns,
10% of monitored columns). The target equivalent uses
`ValueDrift(column="Churn")`, because this Evidently version does not expose
the legacy `TargetDriftPreset`; target drift was **False** with score
**0.011260**. The observed churn-rate shift was **-0.014013**.

Custom metrics include:

- mean MonthlyCharges shift: **19.500625**
- Month-to-month contract-rate shift: **0.244841**
- churn-rate shift: **-0.014013**

The monitoring MLflow run is `d829d6805b764a2693254b70b96871c5` in experiment
`telco-churn-track-a-monitoring`.

Evidence files:

- [`reports/evidently/telco-drift-report.html`](reports/evidently/telco-drift-report.html)
- [`reports/drift-summary.json`](reports/drift-summary.json)
- [`reports/drift-analysis.md`](reports/drift-analysis.md)

The observed feature shifts would trigger investigation of upstream billing and
contract distributions and continued production-performance monitoring. Model
retraining is not automatic; it should be considered only if drift persists and
performance degrades.

Airflow is an optional bonus and is **not implemented**.
