# Location de vélos - Prévoir le nombre de locations

Projet MLOps : prévision du nombre de locations de vélos par heure à partir du jeu de données UCI Bike Sharing, de l'ingestion des données jusqu'au service de prédiction supervisé.

Lien vers le dépôt Git : https://github.com/geoffrey974/5MLDE

## Architecture

```
 data/batches ──simulate──> data/landing ──> Pipeline Prefect (toutes les 60 s)
                                              contrôle qualité -> entraînement -> promotion
                                                       │
                                                       v
                                         MLflow (suivi + registre, alias "champion")
                                                       │  rechargement automatique
                                                       v
 Client ──/predict──> API FastAPI (4 workers) ──> PostgreSQL (prédictions, erreurs, feedback)
                          │ /metrics
                          v
                     Prometheus (règles d'alerte) ──> Grafana (tableau de bord)
```

| Service | Rôle | Adresse |
|---|---|---|
| `api` | prédictions, feedback, métriques | http://127.0.0.1:8000/docs |
| `mlflow` | suivi des expériences et registre de modèles | http://127.0.0.1:5000 |
| `prefect` + `pipeline` | orchestration et déclenchement automatique | http://127.0.0.1:4200 |
| `postgres` | bases `mlflow` et `predictions` | interne |
| `prometheus` | collecte des métriques et alertes | http://127.0.0.1:9090 |
| `grafana` | tableau de bord de supervision | http://127.0.0.1:3000 |

## Démarrage rapide (Docker)

Prérequis : Docker Desktop, et les ports 3000, 4200, 5000, 8000 et 9090 libres.

```
Copy-Item .env.example .env              # puis changer les mots de passe dans .env
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
pip install -e .
python -m bike_mlops.data.ingest         # télécharge les données et crée les 24 lots mensuels
docker compose up -d --build             # lance toute la plateforme
python -m bike_mlops.pipeline.simulate -n 13
```

Le pipeline détecte les lots arrivés, entraîne un modèle et le promeut. En moins d'une minute, l'API sert le nouveau champion sans redémarrage. Grafana : http://127.0.0.1:3000 (identifiants `admin` et `GRAFANA_ADMIN_PASSWORD`).

Arrêt : `docker compose down`. Pour tout effacer, y compris les bases : `docker compose down -v`.

La synthèse d'exploitation (traçabilité, incidents, dimensionnement, limites) est dans [docs/SYNTHESE_OPS.md](docs/SYNTHESE_OPS.md).

## Installation locale (sans Docker)

Prérequis : Python 3.13.

```
python -m venv .venv                    # Créer un environnement
.\.venv\Scripts\Activate                # Lancer cet environnement
pip install -r requirements.txt         # Installer chaque dépendance pour ce projet
pip install -e .                        # Installer le package du projet
Copy-Item .env.example .env             # créer un fichier .env avec la base du fichier .env.example
```

## Données

- **Source** : UCI Machine Learning Repository - Bike Sharing Dataset (https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).
- **Licence** : Creative Commons Attribution 4.0 (CC BY 4.0).
- **Article de référence** : Fanaee-T, H. & Gama, J. (2013). *Event labeling combining ensemble detectors and background knowledge*, Progress in Artificial Intelligence, Springer.
- **Citation du dataset** : Fanaee-T, H. (2013). Bike Sharing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894
- **Contenu utilisé** : `hour.csv`, soit 17 379 relevés horaires de 2011 à 2012 (météo, calendrier, nombre de locations).

Le dataset n'est pas versionné. Il est téléchargé par le script d'ingestion, qui enregistre dans `data/raw/metadata.json` l'URL source, la date de téléchargement et l'empreinte SHA-256 du fichier.

**Simulation de l'arrivée de données** : le dataset historique est découpé en 24 lots mensuels (`data/batches/batch_AAAA-MM.csv`), rejoués successivement pour simuler l'arrivée de nouvelles données.

## Utilisation

### Ingestion
```
python -m bike_mlops.data.ingest
```

### Contrôle qualité d'un lot
```
python -m bike_mlops.data.validate data\batches\batch_2011-01.csv
```

Chaque lot est vérifié avant tout traitement : colonnes attendues, types, plages de valeurs, absence de valeurs manquantes, unicité date/heure et cohérence `cnt = casual + registered`. Un lot non conforme lève une erreur, et la commande renvoie le code 1, ce qui bloque la suite de la chaîne.

### Suivi des expériences (Interface MLflow)
En local, dans un terminal laissé ouvert (inutile avec Docker) :
```
mlflow db upgrade sqlite:///mlflow.db     # première fois uniquement : crée la base
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000
```

Interface : http://127.0.0.1:5000

### Entraînement

```
python -m bike_mlops.models.train --until 2011-12   # s'entraîne sur les lots jusqu'à 2011-12, puis évalue sur 2012-01
python -m bike_mlops.models.train                    # par défaut : tous les lots sauf le dernier, qui sert à l'évaluation
```

Chaque lot est validé avant l'entraînement : un lot non conforme interrompt l'exécution. Chaque run enregistre dans MLflow :
- **paramètres** : les hyperparamètres du modèle, les périodes d'entraînement, le nombre de lots et de lignes et la liste des features.
- **métriques** : RMSE, MAE, et R² sur le mois suivant.
- **données** : empreinte SHA-256 des lots utilisés (tag `data_sha256`).
- **artefact** : le pipeline complet (prétraitement + RandomForest), avec sa signature d'entrée.
- **registre** : chaque modèle est enregistré comme une nouvelle version, avec les tags `train_until`, `eval_batch`, `rmse`, `data_sha256` et `validation_status`.

### Promotion et retour en arrière

La version en service porte l'alias `champion` dans le registre MLflow (`models:/bike-demand@champion`).

```
python -m bike_mlops.models.registry promote 2          # Applique la règle de mise en service à la version 2
python -m bike_mlops.models.registry rollback           # Remet en service le champion précédent
python -m bike_mlops.models.registry rollback --to 1    # Remet en service une version précise (version 1)
```

Avec Docker, les mêmes commandes se lancent dans le conteneur du pipeline : `docker compose exec pipeline python -m bike_mlops.models.registry rollback`.

Règle de promotion, appliquée dans cet ordre :
1. RMSE de la version au-dessus du seuil `promotion.max_rmse` (config) -> **rejetée**
2. Aucun champion en service -> **promue**
3. Données d'entraînement pas plus récentes que celles du champion -> **rejetée** (le champion a pu voir le mois d'évaluation : comparaison faussée)
4. Sinon, les deux versions sont évaluées sur le même lot (mois suivant l'entraînement du challenger) -> **promue** seulement si sa RMSE est plus basse.

Chaque décision est tracée sur la version (tags `promotion_decision` et `promotion_reason`). Une version promue garde le numéro du champion remplacé (`previous_champion`), ce qui permet le retour arrière. La version annulée reçoit le tag `rolled_back`.

### Orchestration (Prefect)

Les lots arrivent un par un dans `data/landing` (simulation de l'arrivée mensuelle des données). Toutes les 60 s (`pipeline.interval_seconds`), le pipeline vérifie s'il y a un nouveau lot et, si oui, enchaîne : contrôle qualité → entraînement → promotion.

Avec Docker, les services `prefect` et `pipeline` tournent en permanence. En local, avec MLflow démarré, dans trois terminaux :
```
prefect server start                                          # interface : http://127.0.0.1:4200
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api  # une seule fois
python -m bike_mlops.pipeline.flow --serve                     # déclenchement automatique
```
```
python -m bike_mlops.pipeline.simulate -n 12    # fait arriver les 12 premiers mois
python -m bike_mlops.pipeline.simulate          # fait arriver le mois suivant
python -m bike_mlops.pipeline.flow              # exécution unique, sans planification
```

Gestion des échecs :
- **reprises automatiques** sur les étapes dépendant de MLflow (indisponibilité passagère) ;
- **pas de reprise** sur le contrôle qualité : un lot invalide bloque la chaîne, aucun modèle n'est entraîné ;
- **une seule exécution à la fois** (`limit=1`) ;
- **alerte** à chaque échec : log `ALERTE`, ligne ajoutée dans `logs/alerts.jsonl`, et message envoyé au webhook `ALERT_WEBHOOK_URL` s'il est défini (Discord, Slack...).

### API de prédiction (FastAPI)

En local, avec MLflow démarré :
```
uvicorn bike_mlops.api.main:app --port 8000
```
Documentation interactive : http://127.0.0.1:8000/docs

| Endpoint | Rôle |
|---|---|
| `GET /health` | état du service (`ok` / `degraded`) et version du modèle chargé |
| `POST /predict` | prédiction du nombre de locations pour une heure donnée |
| `POST /feedback` | enregistre la valeur réellement observée pour une prédiction (`prediction_id`, `actual`) |
| `POST /reload` | recharge immédiatement le champion |
| `GET /metrics` | métriques Prometheus (HTTP et métier) |

Exemple :
```
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"season": 1, "yr": 1, "mnth": 1, "hr": 8, "holiday": 0, "weekday": 2, "workingday": 1, "weathersit": 1, "temp": 0.3, "atemp": 0.3, "hum": 0.5, "windspeed": 0.2}'
```
Réponse : `{"prediction_id": 1, "prediction": 483.08, "model_name": "bike-demand", "model_version": "1"}`

Le modèle servi est celui qui porte l'alias `champion` dans le registre. Chaque réponse indique sa version. Chaque worker vérifie le registre toutes les 30 s (`api.refresh_seconds`) et recharge le modèle si le champion a changé : une promotion ou un retour arrière est pris en compte sans redémarrer l'API.

Chaque appel à `/predict` est enregistré dans la base (`predictions.db` en local, PostgreSQL avec Docker via `DATABASE_URL`) : entrées, prédiction, version du modèle, durée, statut et erreur éventuelle. Les retours terrain sont stockés dans la table `feedback`.

Gestion des erreurs :
- **422** : entrée invalide (champ manquant ou inconnu, mauvais type, valeur hors plage). Les plages sont les mêmes que celles du contrôle qualité. L'erreur est enregistrée.
- **503** : aucun modèle chargé (MLflow indisponible au démarrage). L'API reste joignable et `/health` indique `degraded`.
- **404** : feedback pour une prédiction inconnue.
- **500** : erreur imprévue, avec un message générique. Le détail est uniquement dans les logs.

Une fois le modèle chargé, l'API continue de répondre même si MLflow tombe : le modèle est en mémoire.

### Supervision (Prometheus et Grafana)

Le tableau de bord **Bike demand API - supervision** est provisionné automatiquement (`monitoring/grafana/dashboards/bike-api.json`) : version servie, disponibilité, débit, taux d'erreur, latence p50/p95, prédictions par version, erreur mesurée par le feedback, alertes actives.

Règles d'alerte Prometheus (`monitoring/prometheus/alerts.yml`), visibles sur http://127.0.0.1:9090/alerts :

| Alerte | Condition |
|---|---|
| `ApiIndisponible` | API injoignable depuis 1 min |
| `AucunModeleCharge` | API sans modèle depuis 1 min |
| `LatencePredictElevee` | p95 de `/predict` > 500 ms pendant 2 min |
| `TauxErreurEleve` | plus de 1 % de réponses 5xx pendant 2 min |

### Tests de charge et dimensionnement

```
locust -f loadtests/locustfile.py --host http://127.0.0.1:8000 --headless -u 50 -r 10 -t 60s
```

Résultats, analyse et recommandation : [loadtests/RESULTATS.md](loadtests/RESULTATS.md).

### Tests
```
pytest -v
```

40 tests : configuration, ingestion, contrôle qualité, entraînement, registre (promotion, rejet, retour arrière), orchestration (arrivée des lots, alertes), API (validation, erreurs, persistance, feedback, métriques, rechargement du champion).

## Structure du projet

```
configs/config.yaml          configuration (données, features, modèle, seuils, base)
src/bike_mlops/
  data/                      ingestion et contrôle qualité
  features/                  prétraitement
  models/                    entraînement et registre (promotion, retour arrière)
  pipeline/                  orchestration Prefect et simulation d'arrivée des lots
  api/                       API FastAPI, persistance, métriques
tests/                       tests automatisés (pytest)
loadtests/                   scénario Locust, résultats et dimensionnement
monitoring/                  configuration Prometheus et Grafana
docker/                      initialisation PostgreSQL
docs/                        synthèse OPS
Dockerfile, docker-compose.yml
```
