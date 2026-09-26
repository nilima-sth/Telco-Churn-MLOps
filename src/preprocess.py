"""Leakage-safe preprocessing design for later Track A model experiments."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import ID_COLUMN, TARGET_COLUMN, validate_schema


NUMERIC_COLUMNS = ("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges")
RANDOM_SEED = 42


def coerce_total_charges(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert blank/string TotalCharges to numeric NaN; preserve all other values."""
    result = frame.copy()
    result["TotalCharges"] = pd.to_numeric(result["TotalCharges"].astype("string").str.strip(), errors="coerce")
    return result


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create model inputs and 0/1 target, excluding customer identifier leakage."""
    validate_schema(frame)
    prepared = coerce_total_charges(frame)
    target = prepared[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    if target.isna().any():
        raise ValueError("Churn target could not be encoded.")
    return prepared.drop(columns=[ID_COLUMN, TARGET_COLUMN]), target.astype("int8")


def categorical_columns(features: pd.DataFrame) -> list[str]:
    return [column for column in features.columns if column not in NUMERIC_COLUMNS]


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build an unfitted pipeline; fitting happens on train data only in Phase 1."""
    categories = categorical_columns(features)
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, list(NUMERIC_COLUMNS)),
        ("categorical", categorical_pipeline, categories),
    ])
