# Location de vélos - Prévoir le nombre de locations

Projet MLOps : Prévision du nombre de location de vélos à partir du jeu de données UCI Bike Sharing.

lien vers le repo Git : https://github.com/geoffrey974/5MLDE/tree/develop

## Installation

Prérequis : Python 3.13, Docker

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
- **Contenu utilisé** : 'hour.csv', soit 17 379 relevés horaires de 2011 à 2012 (météo, calendrier, nombre de locations).

Le dataset n'est pas versionné. Il est téléchargé par le script d'ingestion, qui enregistre dans 'data/raw/metadata.json' l'URL source, la date de téléchargement et l'empreinte SHA-256 du fichier.

**Simulation de l'arrivée de données** : le dataset historique est découpé en 24 lots mensuels ('data/batches/batch_AAAA-MM.csv'), rejoués successivement pour simuler l'arrivée de nouvelles données.

## Utilisation

### Ingestion
```
python -m bike_mlops.data.ingest
```

### Contrôle qualité d'un lot
```
python -m bike_mlops.data.validate data\batches\batch_2011-01.csv
```

Chaque lot est vérifié avant tout traitement : colonnes attendues, types, plages de valeurs, absence de valeurs manquantes, unicité date/heure et cohérence 'cnt = casual + registered'. Un lot non conforme lève une erreur, et la commande renvoie le code 1, ce qui bloque la suite de la chaîne.

### Suivi des expériences (Interface MLflow)
Dans un terminal, laissé ouvert : 
```
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
- **données** : empreinte SHA-256 des lots utilisés (tag `data_sha256`)
- **artefact** : le pipeline complet (prétraitement + RandomForest), avec sa signature d'entrée.
- **registre** : chaque modèle est enregistré comme une nouvelle version, avec les tags `train_until`, `eval_batch`, `rmse`, `data_sha256` et `validation_status`.

### Promotion et retour en arrière

La version en service porte l'alias `champion` dans le registre MLflow (`models:/bike-demand@champion`).

```
python -m bike_mlops.models.registry promote 2          # Applique la règle de mise en service à la version 2
python -m bike_mlops.models.registry rollback           # Remet en service le champion précédent
python -m bike_mlops.models.registry rollback --to 1    # Remet en service une version précise (version 1)
```

Règle de promotion, appliquée dans cet ordre :
1. RMSE de la version au-dessus du seuil `promotion.max_rmse` (config) -> **rejetée**
2. Aucun champion en service -> **promue**
3. Données d'entrainement pas plus récentes que celles du champion -> **rejetée** (le champion a pu voir le mois d'évaluation : comparaison faussée)
4. Sinon, les deux versions sont évaluées sur le même lot (mois suivant l'entrainement du challenger) -> **promue** seulement si sa RMSE est plus basse.

Chaque décision est tracée sur la version (tags `promotion_decision` et `promotion_reason`). Une version promue garde le numéro du champion remplacé (`previous_champion`), ce qui permet le retour arrière. La version annulée reçoit le tag `rolled_back`.

### Tests
```
pytest -v
```
