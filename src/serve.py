"""FastAPI serving wrapper for the production alias of the registered model."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from src.registry import MODEL_NAME, PRODUCTION_ALIAS
from src.train import configure_tracking


YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]


class CustomerFeatures(BaseModel):
    """Raw Telco features consumed by the registered preprocessing pipeline."""

    model_config = ConfigDict(extra="forbid")

    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=72)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.model_dump()])


class PredictionResponse(BaseModel):
    prediction: int
    churn: Literal["No", "Yes"]
    churn_probability: float = Field(ge=0, le=1)
    model_name: str
    model_version: str


@lru_cache(maxsize=1)
def load_registered_model():
    """Load the registered production alias, never a local standalone pickle."""
    tracking_uri, _ = configure_tracking()
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    version = client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{PRODUCTION_ALIAS}")
    return model, str(version.version)


def predict_customer(customer: CustomerFeatures) -> PredictionResponse:
    model, version = load_registered_model()
    frame = customer.as_frame()
    prediction = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0][1])
    return PredictionResponse(
        prediction=prediction,
        churn="Yes" if prediction == 1 else "No",
        churn_probability=probability,
        model_name=MODEL_NAME,
        model_version=version,
    )


app = FastAPI(title="Telco Churn Model Service")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures) -> PredictionResponse:
    return predict_customer(customer)
