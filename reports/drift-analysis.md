# Evidently drift analysis

Reference is the deterministic 70% partition (4930 rows); current is the remaining 30% (2113 rows). Only current received synthetic drift.

- Injected numeric drift: `MonthlyCharges` received +20 plus seeded Gaussian noise (std 2). Mean shifted from 64.4877 to 84.3798; custom shift = **19.5006**.
- Injected categorical drift: `Contract=Month-to-month` increased from 0.5390 to 0.7998; custom rate shift = **0.2448**.
- Evidently detected drifted features: **Contract, MonthlyCharges**. Dataset drift detected = **True**.
- Target equivalent: Evidently 0.7.23 has no legacy `TargetDriftPreset`; `ValueDrift(column='Churn')` was used. Churn score = 0.011260; target drift detected = **False**. Churn-rate shift = -0.0140.
- Monitoring MLflow run: `d829d6805b764a2693254b70b96871c5`.

The injected feature drift would warrant investigating the upstream contract and billing distributions, then evaluating production performance. Retraining should be considered only if the shift persists and model performance degrades; this run does not retrain automatically.
