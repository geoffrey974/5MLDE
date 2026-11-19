"""Simule l'arrivée des données : publie les lots mensuels un par un dans le dossier d'arrivée."""
import argparse
import logging
import shutil
from pathlib import Path

from bike_mlops.config import load_config

logger = logging.getLogger(__name__)


def publish_next(batches_dir: str | Path, landing_dir: str | Path, n: int = 1) -> list[Path]:
    """Copie les `n` prochains lots pas encore arrivés. Renvoie les lots publiés."""
    landing = Path(landing_dir)
    landing.mkdir(parents=True, exist_ok=True)
    pending = [p for p in sorted(Path(batches_dir).glob("batch_*.csv")) if not (landing / p.name).exists()]
    published = []
    for p in pending[:n]:
        shutil.copy2(p, landing / p.name)
        published.append(landing / p.name)
        logger.info("Nouveau lot arrivé : %s", p.name)
    if not pending:
        logger.info("Tous les lots sont déjà arrivés")
    return published


def main() -> None:
    """Point d'entrée : publie un ou plusieurs lots."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    parser = argparse.ArgumentParser(description="Simule l'arrivée de nouveaux lots mensuels")
    parser.add_argument("-n", type=int, default=1, help="Nombre de lots à publier (1 par défaut)")
    args = parser.parse_args()
    cfg = load_config()
    publish_next(cfg["data"]["batches_dir"], cfg["data"]["landing_dir"], args.n)


if __name__ == "__main__":
    main()
