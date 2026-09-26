from fastapi.testclient import TestClient

from src import serve


SAMPLE = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes", "tenure": 60,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL", "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes", "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
    "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 65.0, "TotalCharges": 3900.0,
}


class FakeModel:
    def predict(self, frame):
        assert len(frame) == 1
        return [0]

    def predict_proba(self, frame):
        assert len(frame) == 1
        return [[0.82, 0.18]]


def test_health() -> None:
    response = TestClient(serve.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction_uses_raw_features_and_returns_probability(monkeypatch) -> None:
    monkeypatch.setattr(serve, "load_registered_model", lambda: (FakeModel(), "1"))
    response = TestClient(serve.app).post("/predict", json=SAMPLE)

    assert response.status_code == 200
    body = response.json()
    assert body == {"prediction": 0, "churn": "No", "churn_probability": 0.18, "model_name": "telco-churn-classifier", "model_version": "1"}
    assert 0 <= body["churn_probability"] <= 1


def test_registered_loader_uses_production_alias(monkeypatch) -> None:
    requested_uris = []

    class FakeClient:
        def get_model_version_by_alias(self, name, alias):
            assert (name, alias) == ("telco-churn-classifier", "production")
            return type("Version", (), {"version": 2})()

    monkeypatch.setattr(serve, "configure_tracking", lambda: ("sqlite:///test.db", "1"))
    monkeypatch.setattr(serve.mlflow.tracking, "MlflowClient", lambda tracking_uri: FakeClient())
    monkeypatch.setattr(serve.mlflow.sklearn, "load_model", lambda uri: requested_uris.append(uri) or FakeModel())
    serve.load_registered_model.cache_clear()

    _, version = serve.load_registered_model()

    assert version == "2"
    assert requested_uris == ["models:/telco-churn-classifier@production"]
    serve.load_registered_model.cache_clear()


def test_invalid_request_is_rejected() -> None:
    invalid = {**SAMPLE, "SeniorCitizen": 3}
    response = TestClient(serve.app).post("/predict", json=invalid)
    assert response.status_code == 422
