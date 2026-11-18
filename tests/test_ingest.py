"""Tests de l'ingestion"""

import pandas as pd

from bike_mlops.data.ingest import split_into_batches

def test_split_into_batches_creates_one_file_per_month(tmp_path):
    raw = tmp_path / "hour.csv"
    pd.DataFrame({
        "dteday": ["2011-01-15", "2011-01-15", "2011-02-01"],
        "cnt": [1, 2, 3],
    }).to_csv(raw, index=False)

    paths = split_into_batches(raw, tmp_path / "batches")

    assert [p.name for p in paths] == ["batch_2011-01.csv", "batch_2011-02.csv"]
    assert len(pd.read_csv(paths[0])) == 2