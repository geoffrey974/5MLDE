"""Tests de l'entraînement."""

import pytest

from bike_mlops.data.validate import DataValidationError
from bike_mlops.models.train import build_model, evaluate, select_batches, train
from conftest import make_batch

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

    run_id, version = train(cfg, "2011-02")
    run = mlflow.get_run(run_id)

    assert version == "1"

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