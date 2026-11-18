"""Ingestion des donées : Téléchargement du dataset et découpage"""

import hashlib
import io
import json
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from bike_mlops.config import load_config

logger = logging.getLogger(__name__)

def download_raw(cfg: dict) -> Path:
    """Télécharge le dataset et extrait le CSV"""

    raw_dir = Path(cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / cfg["data"]["file"]

    if target.exists():
        # Ne télécharge pas si le fichier existe déjà

        logger.info("Les données brutes sont déjà présentes : %s", target)
        return target

    # Téléchargement du dataset
    url = cfg["data"]["source_url"]
    logger.info("Téléchargement depuis %s", url)
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        target.write_bytes(zf.read(cfg["data"]["file"]))

    metadata = {
        "source_url": url,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "file": target.name,

        # Empreinte du fichier, ce qui permet de savoir si la source du fichier a changé entre 2 ingestions.
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }

    (raw_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Fichier enregistré : %s (sha256=%s)", target, metadata["sha256"])
    return target

def split_into_batches(raw_path: Path, batches_dir: str | Path) -> list[Path]:
    """Découpe du dataset en un fichier CSV par mois (exemple : batch_2011-11)."""
    df = pd.read_csv(raw_path, parse_dates=["dteday"])
    out_dir = Path(batches_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths=[]

    # la period est de 1 mois, pour simuler l'arrivé de nouvelles donées de manière progressive
    for period, group in df.groupby(df["dteday"].dt.to_period("M")):
        path = out_dir / f"batch_{period}.csv"
        group.to_csv(path, index=False)
        paths.append(path)
    return paths

def main() -> None:
    """Lance l'intégralité de l'ingestion"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    cfg = load_config()
    raw_path = download_raw(cfg)
    batches = split_into_batches(raw_path, cfg["data"]["batches_dir"])
    logger.info("%d lots mensuels créés dans %s", len(batches), cfg["data"]["batches_dir"])

if __name__ == "__main__":
    main()