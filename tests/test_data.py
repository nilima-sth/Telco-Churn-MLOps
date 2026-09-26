import pandas as pd
import pytest

from src.data import EXPECTED_COLUMNS, TARGET_COLUMN, load_raw_data, profile_dataset, validate_schema


def test_raw_dataset_matches_expected_schema() -> None:
    frame = load_raw_data()

    assert tuple(frame.columns) == EXPECTED_COLUMNS
    assert len(frame) == 7043
    assert set(frame[TARGET_COLUMN].unique()) == {"Yes", "No"}


def test_profile_reports_known_total_charges_parsing_issue() -> None:
    profile = profile_dataset(load_raw_data())

    assert profile.rows == 7043
    assert profile.columns == 21
    assert profile.total_charges_blank_count == 11
    assert profile.duplicate_rows == 0
    assert profile.duplicate_customer_ids == 0
    assert profile.target_counts == {"No": 5174, "Yes": 1869}


def test_schema_rejects_missing_required_column() -> None:
    frame = load_raw_data().drop(columns=[TARGET_COLUMN])

    with pytest.raises(ValueError, match="missing"):
        validate_schema(frame)
