"""Tests des contrôles qualités des lots"""

import pandas as pd
import pytest

from bike_mlops.data.validate import DataValidationError, validate_batch

@pytest.fixture
def valid_batch() -> pd.DataFrame:
    """Lot conforme a partir des vrai données."""
    return pd.DataFrame({
        "instant": [1, 2],
        "dteday": ["2011-01-01", "2011-01-01"],
        "season": [1, 1],
        "yr": [0, 0],
        "mnth": [1, 1],
        "hr": [0, 1],
        "holiday": [0, 0],
        "weekday": [6, 6],
        "workingday": [0, 0],
        "weathersit": [1, 1],
        "temp": [0.24, 0.22],
        "atemp": [0.2879, .2727],
        "hum": [0.81, 0.80],
        "windspeed": [0.0, 0.0],
        "casual": [3, 8],
        "registered": [13, 32],
        "cnt": [16, 40],
    })

def test_valid_batch_passes(valid_batch):
    result = validate_batch(valid_batch)
    assert len(result) == 2

def test_missing_colum_is_rejected(valid_batch):
    with pytest.raises(DataValidationError):
        validate_batch(valid_batch.drop(columns=["temp"]))

def test_out_of_range_value_is_rejected(valid_batch):
    valid_batch.loc[0, "temp"] = 3.2
    with pytest.raises(DataValidationError):
        validate_batch(valid_batch)

def test_negative_count_is_rejected(valid_batch):
    valid_batch.loc[0, ["casual", "registered", "cnt"]] = [-1, 0, -1]
    with pytest.raises(DataValidationError):
        validate_batch(valid_batch)

def test_inconsistent_total_is_rejected(valid_batch):
    valid_batch.loc[0, "cnt"] = 999
    with pytest.raises(DataValidationError, match="cnt != casual \\+ registered"):
        validate_batch(valid_batch)

def test_missing_value_is_rejected(valid_batch):
    valid_batch.loc[0, "hum"] = None
    with pytest.raises(DataValidationError):
        validate_batch(valid_batch)

def test_duplicate_hour_is_rejected(valid_batch):
    valid_batch.loc[1, "hr"] = 0
    with pytest.raises(DataValidationError):
        validate_batch(valid_batch)

def test_empty_batch_is_rejected(valid_batch):
    with pytest.raises(DataValidationError):
        validate_batch(valid_batch.iloc[0:0])