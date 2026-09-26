import numpy as np

from src.data import load_raw_data
from src.preprocess import coerce_total_charges, make_preprocessor, split_features_target


def test_total_charges_blank_values_become_numeric_missing_values() -> None:
    converted = coerce_total_charges(load_raw_data())

    assert converted["TotalCharges"].dtype.kind == "f"
    assert converted["TotalCharges"].isna().sum() == 11


def test_target_encoding_and_fitted_preprocessor_have_no_missing_values() -> None:
    features, target = split_features_target(load_raw_data())
    transformed = make_preprocessor(features).fit_transform(features)

    assert set(target.unique()) == {0, 1}
    assert "customerID" not in features.columns
    assert transformed.shape[0] == len(features)
    assert not np.isnan(transformed.toarray() if hasattr(transformed, "toarray") else transformed).any()
