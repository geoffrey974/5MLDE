"""Orchestration Prefect : détection d'un nouveau lot, contrôle qualité, entraînement, promotion."""
import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import requests
from mlflow import MlflowClient
from prefect import flow, get_run_logger, task

from bike_mlops.config import get_tracking_uri, load_config
from bike_mlops.data.validate import validate_file
from bike_mlops.models.registry import promote
from bike_mlops.models.train import list_batches, train

logger = logging.getLogger(__name__)


def send_alert(flow, flow_run, state) -> None:
    """Hook d'échec : trace l'alerte dans un fichier et, si configuré, l'envoie sur un webhook."""
    alert = {
        "time": datetime.now(timezone.utc).isoformat(),
        "flow": flow.name,
        "run": flow_run.name,
        "state": state.name,
        "message": state.message,
    }
    path = Path(os.getenv("ALERTS_FILE", "logs/alerts.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(alert, ensure_ascii=False) + "\n")
    logger.error("ALERTE : %s", alert)

    webhook = os.getenv("ALERT_WEBHOOK_URL")
    if webhook:
        try:
            requests.post(webhook, json={"content": f"Échec {flow.name} ({flow_run.name}) : {state.message}"}, timeout=10)
        except requests.RequestException as exc:
            logger.error("Envoi de l'alerte impossible : %s", exc)


@task(retries=3, retry_delay_seconds=10)
def last_trained_month(cfg: dict) -> str | None:
    """Dernier mois d'entraînement présent dans le registre (None si aucune version)."""
    versions = MlflowClient().search_model_versions(f"name='{cfg['model']['name']}'")
    months = [v.tags["train_until"] for v in versions if "train_until" in v.tags]
    return max(months) if months else None


@task
def check_batch(path: Path) -> None:
    """Contrôle qualité bloquant du lot arrivé (pas de retry : un lot invalide le restera)."""
    validate_file(path)


@task(retries=2, retry_delay_seconds=30)
def train_task(cfg: dict, until: str) -> str:
    """Entraîne et enregistre une nouvelle version. Renvoie le numéro de version."""
    _, version = train(cfg, until)
    return version


@task(retries=3, retry_delay_seconds=10)
def promote_task(cfg: dict, version: str) -> bool:
    """Applique la règle de mise en service."""
    return promote(MlflowClient(), cfg, version)


@flow(name="bike-training-pipeline", on_failure=[send_alert], on_crashed=[send_alert])
def training_pipeline() -> str | None:
    """Ré-entraîne seulement si un nouveau lot est arrivé. Renvoie la version créée, sinon None."""
    log = get_run_logger()
    cfg = load_config()
    cfg["data"]["batches_dir"] = cfg["data"]["landing_dir"]
    mlflow.set_tracking_uri(get_tracking_uri(cfg))

    batches = list_batches(cfg["data"]["batches_dir"])
    if len(batches) < 2:
        log.info("Pas assez de lots arrivés (%d) : il en faut 2 (entraînement + évaluation)", len(batches))
        return None

    until = batches[-2].stem.removeprefix("batch_")
    last = last_trained_month(cfg)
    if last is not None and until <= last:
        log.info("Aucun nouveau lot depuis le dernier entraînement (%s) : rien à faire", last)
        return None

    check_batch(batches[-1])
    version = train_task(cfg, until)
    promoted = promote_task(cfg, version)
    log.info("Version %s entraînée jusqu'à %s : %s", version, until, "PROMUE" if promoted else "rejetée")
    return version


def main() -> None:
    """Point d'entrée : une exécution, ou --serve pour un déclenchement automatique périodique."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    parser = argparse.ArgumentParser(description="Pipeline d'entraînement orchestré par Prefect")
    parser.add_argument("--serve", action="store_true", help="Vérifie périodiquement l'arrivée de nouveaux lots")
    args = parser.parse_args()
    if args.serve:
        cfg = load_config()
        # limit = 1 : jamais deux exécutions en parallèle (sinon deux versions identiques)
        training_pipeline.serve(name="bike-training", interval=cfg["pipeline"]["interval_seconds"], limit=1)
    else:
        training_pipeline()


if __name__ == "__main__":
    main()
