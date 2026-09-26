"""Dataset location, loading, and schema validation for Track A."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
EXPECTED_COLUMNS = (
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
)


@dataclass(frozen=True)
class DatasetProfile:
    rows: int
    columns: int
    null_counts: dict[str, int]
    duplicate_rows: int
    duplicate_customer_ids: int
    target_counts: dict[str, int]
    total_charges_blank_count: int
    dtypes: dict[str, str]


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the unmodified IBM CSV and validate its required schema."""
    if not path.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {path}")
    frame = pd.read_csv(path, dtype={"TotalCharges": "string"})
    validate_schema(frame)
    return frame


def validate_schema(frame: pd.DataFrame) -> None:
    missing = set(EXPECTED_COLUMNS) - set(frame.columns)
    unexpected = set(frame.columns) - set(EXPECTED_COLUMNS)
    if missing or unexpected:
        raise ValueError(f"Unexpected Telco schema; missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    invalid_target = set(frame[TARGET_COLUMN].dropna().unique()) - {"Yes", "No"}
    if invalid_target:
        raise ValueError(f"Unexpected Churn labels: {sorted(invalid_target)}")


def profile_dataset(frame: pd.DataFrame) -> DatasetProfile:
    """Return transparent Phase 0 diagnostics without modifying raw data."""
    validate_schema(frame)
    total_charges = frame["TotalCharges"].astype("string")
    blank_total_charges = int(total_charges.str.strip().eq("").sum())
    return DatasetProfile(
        rows=len(frame),
        columns=len(frame.columns),
        null_counts={name: int(value) for name, value in frame.isna().sum().items()},
        duplicate_rows=int(frame.duplicated().sum()),
        duplicate_customer_ids=int(frame[ID_COLUMN].duplicated().sum()),
        target_counts={str(key): int(value) for key, value in frame[TARGET_COLUMN].value_counts().items()},
        total_charges_blank_count=blank_total_charges,
        dtypes={name: str(dtype) for name, dtype in frame.dtypes.items()},
    )
