"""Tests du chargement de la configuration"""

import pytest

from bike_mlops.config import load_config

def test_default_config_has_data_section():
    assert "source_url" in load_config()["data"]

def test_empty_config_raises(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="vide"):
        load_config(empty)