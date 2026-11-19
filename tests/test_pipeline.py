"""Tests de l'orchestration : arrivée des lots et alertes."""

import json
from types import SimpleNamespace

from bike_mlops.pipeline.flow import send_alert
from bike_mlops.pipeline.simulate import publish_next
from conftest import make_batch


def test_publish_next_copies_batches_in_order(tmp_path):
    archive, landing = tmp_path / "batches", tmp_path / "landing"
    archive.mkdir()
    for month in ["2011-01", "2011-02", "2011-03"]:
        make_batch(month).to_csv(archive / f"batch_{month}.csv", index=False)

    assert [p.name for p in publish_next(archive, landing, 2)] == ["batch_2011-01.csv", "batch_2011-02.csv"]
    assert [p.name for p in publish_next(archive, landing)] == ["batch_2011-03.csv"]
    assert publish_next(archive, landing) == []


def test_send_alert_writes_alert_file(tmp_path, monkeypatch):
    alerts = tmp_path / "alerts.jsonl"
    monkeypatch.setenv("ALERTS_FILE", str(alerts))
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)

    send_alert(
        SimpleNamespace(name="bike-training-pipeline"),
        SimpleNamespace(name="run-test"),
        SimpleNamespace(name="Failed", message="lot invalide"),
    )

    alert = json.loads(alerts.read_text(encoding="utf-8").splitlines()[0])
    assert alert["state"] == "Failed"
    assert alert["message"] == "lot invalide"
