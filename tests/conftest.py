"""Données et fixtures communes aux tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bike_mlops.config import load_config

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