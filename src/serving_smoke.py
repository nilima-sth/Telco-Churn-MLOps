"""Record real local serving checks for the production registered model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

from src.data import PROJECT_ROOT


EVIDENCE_PATH = PROJECT_ROOT / "reports" / "serving-smoke-test.json"

LOWER_RISK_CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes", "tenure": 60,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL", "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes", "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
    "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 65.0, "TotalCharges": 3900.0,
}

HIGHER_RISK_CUSTOMER = {
    "gender": "Male", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No", "tenure": 2,
    "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "Fiber optic", "OnlineSecurity": "No",
    "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 99.0, "TotalCharges": 198.0,
}


def _request_json(url: str, method: str = "GET", body: dict | None = None) -> dict:
    encoded = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(url, data=encoded, method=method, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def run_smoke_test(base_url: str) -> dict:
    base_url = base_url.rstrip("/")
    evidence = {
        "health": _request_json(f"{base_url}/health"),
        "predictions": [
            {"case": "lower-risk-style", "request": LOWER_RISK_CUSTOMER, "response": _request_json(f"{base_url}/predict", "POST", LOWER_RISK_CUSTOMER)},
            {"case": "higher-risk-style", "request": HIGHER_RISK_CUSTOMER, "response": _request_json(f"{base_url}/predict", "POST", HIGHER_RISK_CUSTOMER)},
        ],
    }
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Run two live Track A serving smoke tests.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Running FastAPI service URL.")
    args = parser.parse_args()
    evidence = run_smoke_test(args.url)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
