import pandas as pd

from src.data import load_raw_data
from src import drift
from src.drift import custom_metrics, inject_drift, make_reference_current, prepare_monitoring_data


def test_reference_current_split_is_deterministic() -> None:
    frame = load_raw_data()
    first = make_reference_current(frame)
    second = make_reference_current(frame)
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    assert len(first[0]) == 4930
    assert len(first[1]) == 2113


def test_injection_only_changes_current() -> None:
    frame = load_raw_data()
    reference, current = make_reference_current(frame)
    original_reference = reference.copy(deep=True)
    result = inject_drift(reference, current)

    pd.testing.assert_frame_equal(reference, original_reference)
    assert result.current["MonthlyCharges"].mean() > result.before_current["MonthlyCharges"].mean() + 15
    assert result.current["Contract"].value_counts(normalize=True)["Month-to-month"] > result.before_current["Contract"].value_counts(normalize=True)["Month-to-month"]


def test_custom_metrics_are_real_numeric_values() -> None:
    data = prepare_monitoring_data(load_raw_data())
    metrics = custom_metrics(data)
    assert metrics["mean_monthly_charges_shift"] > 15
    assert metrics["month_to_month_rate_shift"] > 0
    assert isinstance(metrics["churn_rate_shift"], float)


def test_monitoring_summary_and_report_are_created(tmp_path, monkeypatch) -> None:
    data = prepare_monitoring_data(load_raw_data())
    report_path = tmp_path / "telco-drift-report.html"
    summary_path = tmp_path / "drift-summary.json"
    monkeypatch.setattr(drift, "REPORT_DIRECTORY", tmp_path)
    monkeypatch.setattr(drift, "REPORT_PATH", report_path)
    monkeypatch.setattr(drift, "SUMMARY_PATH", summary_path)

    summary = drift.run_monitoring(data)

    assert report_path.exists()
    assert summary_path.exists()
    assert summary["data_drift"]["drifted_features"] == ["Contract", "MonthlyCharges"]
    assert set(summary["custom_metrics"]) >= {"mean_monthly_charges_shift", "churn_rate_shift"}
