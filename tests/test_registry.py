"""Tests du registre : promotion, rejet et retour en arrière."""

import pytest
from mlflow import MlflowClient

from bike_mlops.models.registry import get_champion, promote, rollback
from bike_mlops.models.train import train

@pytest.fixture
def versions(cfg) -> MlflowClient:
    """Deux versions : v1 entrainée jusqu'à 2011-01, v2 jusqu'à 2011-02."""
    cfg["promotion"]["max_rmse"] = 1000
    train(cfg, "2011-01")
    train(cfg, "2011-02")
    return MlflowClient()

def test_first_version_is_promoted(cfg, versions):
    assert promote(versions, cfg, "1") is True
    assert str(get_champion(versions, cfg["model"]["name"]).version) == "1"

def test_version_above_threshold_is_rejected(cfg, versions):
    cfg["promotion"]["max_rmse"] = 0
    assert promote(versions, cfg, "1") is False
    assert get_champion(versions, cfg["model"]["name"]) is None
    tags = versions.get_model_version(cfg["model"]["name"], "1").tags
    assert tags["promotion_decision"] == "rejected"

def test_older_data_is_rejected(cfg, versions):
    promote(versions, cfg, "2")
    assert promote(versions, cfg, "1") is False
    assert str(get_champion(versions, cfg["model"]["name"]).version) == "2"

def test_rollback_to_given_version(cfg, versions):
    name = cfg["model"]["name"]
    promote(versions, cfg, "2")
    assert rollback(versions, cfg, "1") == "1"
    assert str(get_champion(versions, name).version) == "1"
    assert versions.get_model_version(name, "2").tags["rolled_back"] == "true"

def test_rollback_without_previous_champion_fails(cfg, versions):
    promote(versions, cfg, "1")
    with pytest.raises(ValueError):
        rollback(versions, cfg)
