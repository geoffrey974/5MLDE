from pathlib import Path

import yaml
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

def load_config(path: str | Path | None = None) -> dict:
    """Charge le fichier config.yaml"""

    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not cfg:
        raise ValueError(f"Fichier de configuration vide : {config_path}")
    return cfg

def get_tracking_uri(cfg: dict) -> str:
    """Adresse du server MLflow : la variable d'environement est prioritaire sur la config"""
    return os.getenv("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])