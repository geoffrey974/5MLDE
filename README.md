# Location de vélos - Prévoir le nombre de locations

Projet MLOps : Prévision du nombre de location de vélos à partir du jeu de données UCI Bike Sharing.

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


### Tests
```
pytest -v
```