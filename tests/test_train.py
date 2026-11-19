"""Tests de l'entraînement."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bike_mlops.config import load_config
from bike_mlops.data.validate import DataValidationError
from bike_mlops.models.train import build_model, evaluate,select_batches,train

def make_batch(month: str, n_hours: int = 48, seed: int = 0) -> pd.DataFrame:
    """Génère un lot synthétique conforme au schéma."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(f"{month}-01", periods=n_hours, freq="h")
    casual = rng.integers(0, 50, n_hours)
    registered = rng.integers(0,200, n_hours)
    return pd.DataFrame({
        "instant": range(1, n_hours + 1),
        "dteday": dates.strftime("%Y-%m-%d"),
        "season": 1,
        "yr": 0,
        "mnth": dates.month,
        "hr": dates.hour,
        "holiday": 0,
        "weekday": dates.day_of_week,
        "workingday": 1,
        "weathersit": 1,
        "temp": rng.uniform(0, 1, n_hours),
        "atemp": rng.uniform(0, 1, n_hours),
        "hum": rng.uniform(0, 1, n_hours),
        "windspeed": rng.uniform(0, 1, n_hours),
        "casual": casual,
        "registered": registered,
        "cnt": casual + registered,
    })

@pytest.fixture
def batches_dir(tmp_path: Path) -> Path:
    """Dossier contenant 3 lots mensuels synthétiques."""
    d = tmp_path / "batches"
    d.mkdir()
    for i, month in enumerate(["2011-01", "2011-02", "2011-03"]):
        make_batch(month, seed=i).to_csv(d / f"batch_{month}.csv", index=False)
    return d

@pytest.fixture
def cfg(batches_dir: Path, tmp_path: Path, monkeypatch) -> dict:
    """Config du projet pointant vers les lots de test et un MLflow local temporaire."""
    config = load_config()
    config["data"]["batches_dir"] = str(batches_dir)
    config["model"]["params"]["n_estimators"] = 5

    # MLflow de test isolé avec base SQLite et artefacts dans le dossier temporaire
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    return config

def test_select_batches_uses_next_month_for_evaluation(batches_dir):
    batches = sorted(batches_dir.glob("batch_*.csv"))
    train_paths, eval_path = select_batches(batches, "2011-01")
    assert [p.name for p in train_paths] == ["batch_2011-01.csv"]
    assert eval_path.name == "batch_2011-02.csv"

def test_select_batches_default_keeps_last_month_for_evaluation(batches_dir):
    batches = sorted(batches_dir.glob("batch_*.csv"))
    train_paths, eval_path = select_batches(batches, None)
    assert len(train_paths) == 2
    assert eval_path.name == "batch_2011-03.csv"

def test_select_batches_rejects_last_month(batches_dir):
    batches = sorted(batches_dir.glob("batch_*.csv"))
    with pytest.raises(ValueError):
        select_batches(batches, "2011-03")

def test_evaluate_returns_expected_metrics(cfg):
    df = make_batch("2011-01")
    X, y = df[cfg["features"]["numeric"] + cfg["features"]["categorical"]], df["cnt"]
    model = build_model(cfg).fit(X,y)
    assert set(evaluate(model, X, y)) == {"rmse", "mae", "r2"}

def test_train_logs_run_in_mlflow(cfg):
    import mlflow

    run_id = train(cfg, "2011-02")
    run = mlflow.get_run(run_id)

    assert run.info.status == "FINISHED"
    assert "rmse" in run.data.metrics
    assert run.data.params["train_until"] == "2011-02"
    assert "data_sha256" in run.data.tags

def test_train_stops_on_invalid_batch(cfg, batches_dir):
    bad = make_batch("2011-02", seed=1)
    bad.loc[0, "temp"] = 3.2
    bad.to_csv(batches_dir / "batch_2011-02.csv", index=False)
    with pytest.raises(DataValidationError):
        train(cfg, "2011-02")