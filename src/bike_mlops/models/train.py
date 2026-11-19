"""Entrainement du modèle et suivi dans MLflow (paramètres, métriques, données, artefacts)."""
import argparse
import hashlib
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from mlflow.models import infer_signature
from mlflow import MlflowClient

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from bike_mlops.config import load_config, get_tracking_uri
from bike_mlops.data.validate import validate_file
from bike_mlops.features.preprocess import build_preprocessor, split_features_target
from bike_mlops.models.registry import tag_version

logger = logging.getLogger(__name__)

def list_batches(batches_dir: str | Path) -> list[Path]:
    """Liste des lots disponibles triés par mois."""
    return sorted(Path(batches_dir).glob("batch_*.csv"))

def select_batches(batches: list[Path], until: str | None) -> tuple[list[Path], Path]:
    """Lots d'entrainement (jusqu'au mois `until` inclus), et les lots d'évaluation (mois suivant)."""
    months = [p.stem.removeprefix("batch_") for p in batches]
    if until is None:
        idx = len(batches) - 2
    elif until in months:
        idx = months.index(until)
    else:
        raise ValueError(f"Aucun lot pour le mois {until}")
    if idx < 0 or idx + 1 >= len(batches):
        raise ValueError(f"Pas de lot disponible après {months[idx] if idx >= 0 else until} pour l'évaluation")
    return batches[: idx + 1], batches[idx + 1]

def load_validated(paths: list[Path]) -> pd.DataFrame:
    """Charge et valide chaque lot, un lot non conforme interrompt l'entrainement."""
    return pd.concat([validate_file(p) for p in paths], ignore_index=True)

def build_model(cfg: dict) -> Pipeline:
    """Pipeline complet : Prétraitement + modèle, enregistré d'un seul bloc dans MLflow."""
    return Pipeline([
        ("preprocess", build_preprocessor(cfg)),
        ("model", RandomForestRegressor(**cfg["model"]["params"])),
    ])

def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Calcule les métriques de régression sur un jeu d'évaluation."""
    pred = model.predict(X)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }

def data_fingerprint(paths: list[Path]) -> str:
    """Empreinte SHA-256 des lots utilisés. Identifie exactement les données d'un run."""
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()

def train(cfg: dict, until: str | None = None) -> tuple[str, str]:
    """Entraîne, évalue et enregistre le modèle dans MLflow et le registre. Renvoie `run_id` et `version`"""
    train_paths, eval_path = select_batches(list_batches(cfg["data"]["batches_dir"]), until)
    train_df = load_validated(train_paths)
    eval_df = load_validated([eval_path])
    X_train, y_train = split_features_target(train_df, cfg)
    X_eval, y_eval = split_features_target(eval_df, cfg)

    # La variable d'environnement est prioritaire sur la config
    mlflow.set_tracking_uri(get_tracking_uri(cfg))

    train_until = train_paths[-1].stem.removeprefix("batch_")
    with mlflow.start_run(run_name=f"train-until-{train_until}") as run:
        model = build_model(cfg)
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_eval, y_eval)

        mlflow.log_params(cfg["model"]["params"])
        mlflow.log_params({
            "train_until": train_until,
            "eval_batch": eval_path.stem,
            "n_train_batches": len(train_paths),
            "n_train_rows": len(train_df),
            "n_eval_rows": len(eval_df),
            "features": ",".join(X_train.columns),
        })

        data_sha256 = data_fingerprint(train_paths)
        mlflow.set_tag("data_sha256", data_sha256)
        mlflow.log_metrics(metrics)
        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X_eval, model.predict(X_eval)),
            input_example=X_eval.head(3),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
            registered_model_name=cfg["model"]["name"],
        )

        version = str(model_info.registered_model_version)
        tag_version(MlflowClient(), cfg["model"]["name"], version, {
            "train_until": train_until,
            "eval_batch": eval_path.stem,
            "rmse": round(metrics["rmse"], 3),
            "data_sha256": data_sha256,
            "validation_status": "passed",
        })
        logger.info("Run %s -> Version %s : %s", run.info.run_id, version, metrics)
        return run.info.run_id, version
    
def main() -> None:
    """Point d'entrée"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    parser = argparse.ArgumentParser(description="Entraîne le modèle sur les lots disponibles")
    parser.add_argument("--until", help="Dernier mois d'entrainement (AAAA-MM). Par défaut c'est l'avant-dernier lot.")
    args = parser.parse_args()
    train(load_config(), args.until)

if __name__ == "__main__":
    main()