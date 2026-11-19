"""Model Registry : Versions, tags, règle de promotion et retour en arrière."""

import logging
import argparse

from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from sklearn.metrics import root_mean_squared_error

from bike_mlops.config import get_tracking_uri, load_config
from bike_mlops.data.validate import validate_file
from bike_mlops.features.preprocess import split_features_target

logger = logging.getLogger(__name__)

# Alias de la version en service
CHAMPION = "champion"

def tag_version(client: MlflowClient, name: str, version: str, tags: dict) -> None:
    """Ajout des tags à une version du modèle (période, métriques, empreinte des données...)"""

    for key, value in tags.items():
        client.set_model_version_tag(name, version, key, str(value))

def get_champion(client: MlflowClient, name: str) -> ModelVersion | None:
    """Renvoie la version actuellement en service, ou None s'il n'y en a pas encore."""
    try:
        return client.get_model_version_by_alias(name, CHAMPION)
    except MlflowException:
        return None

def rmse_on_batch(cfg: dict, version: str, batch_path: Path) -> float:
    """Recharge une version depuis le registre et calcule sa RMSE sur un lot validé."""
    model = mlflow.sklearn.load_model(f"models:/{cfg['model']['name']}/{version}")
    X, y = split_features_target(validate_file(batch_path), cfg)
    return float(root_mean_squared_error(y, model.predict(X)))

def promote(client: MlflowClient, cfg: dict, version: str) -> bool:
    """Applique la règle de mise en service à une version. Renvoie True si elle devient champion"""
    name = cfg["model"]["name"]
    max_rmse = cfg["promotion"]["max_rmse"]
    challenger = client.get_model_version(name, version)
    eval_batch = challenger.tags["eval_batch"]
    batch_path = Path(cfg["data"]["batches_dir"]) / f"{eval_batch}.csv"
    challenger_rmse = rmse_on_batch(cfg, version, batch_path)
    champion = get_champion(client, name)

    if challenger_rmse > max_rmse:
        promoted, reason = False, f"RMSE {challenger_rmse:.2f} au dessus du seuil {max_rmse}"
    elif champion is None:
        promoted, reason = True, "Aucun champion au service."
    elif str(champion.version) == str(version):
        promoted, reason = False, "Déjà champion"
    elif challenger.tags["train_until"] <= champion.tags["train_until"]:
        # Le champion a pu être entraîné sur le mois d'évaluation, la comparaison serait donc faussée.
        promoted, reason = False, f"Données pas plus récentes que le champion v{champion.version}"
    else:
        # comparaison équitable, les deux versions sont évaluées sur le même lot
        champion_rmse = rmse_on_batch(cfg, champion.version, batch_path)
        promoted = challenger_rmse < champion_rmse
        verdict = "meilleure" if promoted else "pas meilleure"
        reason = (f"RMSE {challenger_rmse:.2f} {verdict} que v{champion.version} " f"({champion_rmse:.2f}) sur {eval_batch}")

    tag_version(client, name, version, {
        "promotion_decision": "promoted" if promoted else "rejected",
        "promotion_reason": reason,
    })
    if promoted:
        if champion is not None:
            # Mémorise le champion remplacé pour pouvoir revenir en arrière
            tag_version(client, name, version, {"previous_champion": champion.version})
        client.set_registered_model_alias(name, CHAMPION, str(version))
    logger.info("Version %s %s : %s", version, "PROMUE" if promoted else "REJETÉE", reason)

    return promoted

def rollback(client: MlflowClient, cfg: dict, to_version: str | None = None) -> str:
    """Remet en service le champion précédent, ou une version donnée. Renvoie la version choisie."""
    name = cfg["model"]["name"]
    current = get_champion(client, name)
    if current is None:
        raise ValueError("Aucun champion en service : rien à annuler")
    target = to_version or current.tags.get("previous_champion")
    if not target:
        raise ValueError(f"La version {current.version} n'a pas de version précédente : préciser --to")
    client.get_model_version(name, target)
    client.set_registered_model_alias(name, CHAMPION, str(target))
    tag_version(client, name, current.version, {"rolled_back": "true"})
    logger.info("Retour arrière : champion v%s -> v%s", current.version, target)

    return str(target)

def main() -> None:
    """Point d'entrée : Promote version ou rollback"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    parser = argparse.ArgumentParser(description="Gestion du registre de modèles")
    sub = parser.add_subparsers(dest="command", required=True)
    p_promote = sub.add_parser("promote", help="Applique la règle de mise en service à une version")
    p_promote.add_argument("version")
    p_rollback = sub.add_parser("rollback", help="Remet en service le champion précédent")
    p_rollback.add_argument("--to", help="Version à mettre en service")
    args = parser.parse_args()

    cfg = load_config()
    mlflow.set_tracking_uri(get_tracking_uri(cfg))
    client = MlflowClient()
    if args.command == "promote":
        promote(client, cfg, args.version)
    else:
        rollback(client, cfg, args.to)

if __name__ == "__main__":
    main()