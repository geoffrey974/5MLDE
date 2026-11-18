from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

def load_config(path: str | Path | None = None) -> dict:
    """Charge le fichier config.yaml"""

    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Un fichier vide renvoie None
    if not cfg:
        raise ValueError(f"Fichier de configuration vide : {config_path}")
    return cfg