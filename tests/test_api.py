"""Tests de l'API, avec un faux modèle (pas besoin de MLflow)."""

import pytest
from fastapi.testclient import TestClient

from bike_mlops.api.main import app, service

VALID = {
    "season": 1, "yr": 1, "mnth": 1, "hr": 8, "holiday": 0, "weekday": 2, "workingday": 1,
    "weathersit": 1, "temp": 0.3, "atemp": 0.3, "hum": 0.5, "windspeed": 0.2,
}


class FakeModel:
    def predict(self, X):
        return [123.456]


@pytest.fixture
def client(monkeypatch):
    """Client sans démarrage (lifespan) : on injecte directement un faux modèle."""
    monkeypatch.setattr(service, "model", FakeModel())
    monkeypatch.setattr(service, "version", "7")
    return TestClient(app)


def test_predict_returns_prediction_and_version(client):
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    assert response.json() == {"prediction": 123.46, "model_name": "bike-demand", "model_version": "7"}


@pytest.mark.parametrize("field, value", [("hr", 24), ("temp", 1.5), ("weathersit", 0), ("hr", "midi")])
def test_predict_rejects_invalid_input(client, field, value):
    assert client.post("/predict", json={**VALID, field: value}).status_code == 422


def test_predict_rejects_missing_and_unknown_fields(client):
    missing = {k: v for k, v in VALID.items() if k != "temp"}
    assert client.post("/predict", json=missing).status_code == 422
    assert client.post("/predict", json={**VALID, "cnt": 10}).status_code == 422


def test_predict_without_model_returns_503(client, monkeypatch):
    monkeypatch.setattr(service, "model", None)
    assert client.post("/predict", json=VALID).status_code == 503
    assert client.get("/health").json()["status"] == "degraded"


def test_health_reports_model_version(client):
    assert client.get("/health").json() == {"status": "ok", "model_name": "bike-demand", "model_version": "7"}
