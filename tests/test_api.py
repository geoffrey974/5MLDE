"""Tests de l'API, avec un faux modèle (pas besoin de MLflow) et une base SQLite temporaire."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from bike_mlops.api import main
from bike_mlops.api.main import app, service
from bike_mlops.api.storage import Feedback, PredictionLog, PredictionStore

VALID = {
    "season": 1, "yr": 1, "mnth": 1, "hr": 8, "holiday": 0, "weekday": 2, "workingday": 1,
    "weathersit": 1, "temp": 0.3, "atemp": 0.3, "hum": 0.5, "windspeed": 0.2,
}


class FakeModel:
    def predict(self, X):
        return [123.456]


@pytest.fixture
def store(tmp_path, monkeypatch) -> PredictionStore:
    """Base SQLite vide, propre à chaque test."""
    test_store = PredictionStore(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setattr(main, "store", test_store)
    return test_store


@pytest.fixture
def client(store, monkeypatch):
    """Client sans démarrage (lifespan) : on injecte directement un faux modèle."""
    monkeypatch.setattr(service, "model", FakeModel())
    monkeypatch.setattr(service, "version", "7")
    return TestClient(app)


def logs(store: PredictionStore) -> list[PredictionLog]:
    with Session(store.engine) as session:
        return list(session.scalars(select(PredictionLog)))


def test_predict_returns_prediction_and_version(client):
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    assert response.json() == {"prediction_id": 1, "prediction": 123.46, "model_name": "bike-demand", "model_version": "7"}


def test_predict_is_logged_with_version_and_duration(client, store):
    client.post("/predict", json=VALID)
    [row] = logs(store)
    assert (row.status, row.model_version, row.prediction) == ("ok", "7", 123.46)
    assert row.features == VALID
    assert row.duration_ms >= 0


@pytest.mark.parametrize("field, value", [("hr", 24), ("temp", 1.5), ("weathersit", 0), ("hr", "midi")])
def test_predict_rejects_invalid_input(client, field, value):
    assert client.post("/predict", json={**VALID, field: value}).status_code == 422


def test_invalid_input_is_logged_as_error(client, store):
    client.post("/predict", json={**VALID, "hr": 24})
    [row] = logs(store)
    assert row.status == "error"
    assert "hr" in row.error


def test_predict_rejects_missing_and_unknown_fields(client):
    missing = {k: v for k, v in VALID.items() if k != "temp"}
    assert client.post("/predict", json=missing).status_code == 422
    assert client.post("/predict", json={**VALID, "cnt": 10}).status_code == 422


def test_predict_without_model_returns_503_and_is_logged(client, store, monkeypatch):
    monkeypatch.setattr(service, "model", None)
    assert client.post("/predict", json=VALID).status_code == 503
    assert client.get("/health").json()["status"] == "degraded"
    [row] = logs(store)
    assert (row.status, row.error) == ("error", "Aucun modèle chargé")


def test_feedback_is_stored(client, store):
    prediction_id = client.post("/predict", json=VALID).json()["prediction_id"]
    assert client.post("/feedback", json={"prediction_id": prediction_id, "actual": 130}).status_code == 201
    with Session(store.engine) as session:
        [fb] = session.scalars(select(Feedback)).all()
    assert (fb.prediction_id, fb.actual) == (prediction_id, 130)


def test_feedback_for_unknown_prediction_returns_404(client):
    assert client.post("/feedback", json={"prediction_id": 999, "actual": 10}).status_code == 404


def metric(client, line_start: str) -> float:
    """Valeur d'une série lue sur /metrics, 0 si elle n'existe pas encore."""
    for line in client.get("/metrics").text.splitlines():
        if line.startswith(line_start + " "):
            return float(line.split()[-1])
    return 0.0


def test_metrics_endpoint_exposes_http_metrics(client):
    client.get("/health")
    body = client.get("/metrics").text
    assert "http_requests_total" in body
    assert "http_request_duration_seconds_bucket" in body


def test_predictions_are_counted_by_version_and_status(client):
    ok = 'bike_predictions_total{model_version="7",status="ok"}'
    error = 'bike_predictions_total{model_version="7",status="error"}'
    ok_before, error_before = metric(client, ok), metric(client, error)
    client.post("/predict", json=VALID)
    client.post("/predict", json={**VALID, "hr": 24})
    assert metric(client, ok) == ok_before + 1
    assert metric(client, error) == error_before + 1


def test_feedback_error_is_measured(client):
    count = 'bike_feedback_abs_error_count{model_version="7"}'
    total = 'bike_feedback_abs_error_sum{model_version="7"}'
    count_before, total_before = metric(client, count), metric(client, total)
    prediction_id = client.post("/predict", json=VALID).json()["prediction_id"]
    client.post("/feedback", json={"prediction_id": prediction_id, "actual": 133.46})
    assert metric(client, count) == count_before + 1
    assert metric(client, total) == pytest.approx(total_before + 10)


def test_health_reports_model_version(client):
    assert client.get("/health").json() == {"status": "ok", "model_name": "bike-demand", "model_version": "7"}
