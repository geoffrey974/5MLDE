"""Vérification que chaque lot est au bon format et qu'il est bien conforme avant de pouvoir le traiter"""

import argparse
import logging
import sys

from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """Levée quand un lot ne respecte pas le schéma attendu."""

SCHEMA = pa.DataFrameSchema(
    columns={
        "dteday": pa.Column(pa.DateTime, coerce=True),
        "season": pa.Column(int, pa.Check.isin([1,2,3,4])),
        "yr": pa.Column(int, pa.Check.isin([0,1])),
        "mnth": pa.Column(int, pa.Check.in_range(1,12)),
        "hr": pa.Column(int, pa.Check.in_range(0,23)),
        "holiday": pa.Column(int, pa.Check.isin([0,1])),
        "weekday": pa.Column(int, pa.Check.in_range(0,6)),
        "workingday": pa.Column(int, pa.Check.isin([0,1])),
        "weathersit": pa.Column(int, pa.Check.in_range(1,4)),
        "temp": pa.Column(float, pa.Check.in_range(0,1)),
        "atemp": pa.Column(float, pa.Check.in_range(0,1)),
        "hum": pa.Column(float, pa.Check.in_range(0,1)),
        "windspeed": pa.Column(float, pa.Check.in_range(0,1)),
        "casual": pa.Column(int, pa.Check.ge(0)),
        "registered": pa.Column(int, pa.Check.ge(0)),
        "cnt": pa.Column(int, pa.Check.ge(0)),
    },
    checks=[
        pa.Check(lambda df: len(df) > 0, error = "Le lot est vide"),
        pa.Check(lambda df: df["cnt"] == df["casual"] + df["registered"], error = "cnt != casual + registered"),
    ],

    # Une seule ligne par date et par heure
    unique=["dteday", "hr"],
)

def validate_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Valide un lot. Lève DataValidationError avec le détail des anomalies si non conforme."""

    try:
        return SCHEMA.validate(df, lazy=True)
    except SchemaErrors as err:
        failures = err.failure_cases.copy()

        # Pandera rapport un contrôle sur la ligne entière colonne par colonne : on le regroupe en une seule entrée.
        failures.loc[failures["schema_context"] == "DataFrameSchema", "column"] = "(ligne)"
        summary = (
            failures.drop_duplicates(["column", "check", "index"])
            .groupby(["column", "check"], dropna=False)
            .size()
            .reset_index(name="nb_lignes")
        )
        raise DataValidationError(
            f"{len(summary)} anomalie(s) détectée(s) :\n{summary.to_string(index=False)}"
        ) from err

def validate_file(path: str | Path) -> pd.DataFrame:
    """Charge un fichier CSV, puis le valide"""
    logger.info("Validation du lot %s", path)
    df = validate_batch(pd.read_csv(path))
    logger.info("Lot conforme : %s (%d lignes)", path, len(df))
    return df

def main() -> None:
    """Valide si un lot est conforme et renvoie 1 si il n'est pas conforme."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    parser = argparse.ArgumentParser(description="Valide un lot de données")
    parser.add_argument("path", help="Chemin du fichier CSV à Valider")
    args = parser.parse_args()
    try:
        validate_file(args.path)
    except DataValidationError as err:
        logger.error("Lot NON conforme, arrêt : %s", err)
        sys.exit(1)

if __name__ == "__main__":
    main()