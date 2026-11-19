# Tests de charge et dimensionnement

## Protocole

- **Outil** : Locust 2.47 (`loadtests/locustfile.py`), en mode headless, 60 s par palier.
- **Scénario par utilisateur** : 10 prédictions pour 2 retours terrain (`/feedback`) et 1 `/health`, avec une pause de 0,5 à 1,5 s entre deux actions. Les entrées sont tirées au hasard dans les plages valides.
- **Modèle servi** : `bike-demand` v3 (champion), RandomForest de 100 arbres.
- **Base** : SQLite (`predictions.db`). Chaque appel à `/predict` y écrit une ligne.
- **Machine** : Intel Core i5-9300H (4 cœurs / 8 threads), 40 Go de RAM, Windows 11. Locust, l'API et MLflow tournent sur la même machine.
- **Résultats bruts** : `loadtests/results/*_stats.csv` et `*_failures.csv`.

Commande utilisée pour chaque palier :
```
locust -f loadtests/locustfile.py --host http://127.0.0.1:8000 --headless -u 50 -r 10 -t 60s --csv loadtests/results/load_50u
```

## Résultats

### Un seul processus (`uvicorn ... --port 8000`)

| Utilisateurs | Débit (req/s) | Médiane | p95 | Max | Erreurs |
|---|---|---|---|---|---|
| 10 | 9,3 | 42 ms | 110 ms | 210 ms | 0 % |
| 20 | 17,2 | 80 ms | 350 ms | 3,1 s | 0 % |
| 30 | 17,0 | 610 ms | 1,6 s | 3,5 s | 0 % |
| 50 | 28,7 | 560 ms | 1,8 s | 5,9 s | 0 % |
| 100 | 16,7 | 4,5 s | 9,4 s | 12 s | 0 % |
| 200 | 15,2 | 11 s | 17 s | 29 s | 0 % |

### Quatre processus (`uvicorn ... --workers 4`)

| Utilisateurs | Débit (req/s) | Médiane | p95 | Max | Erreurs |
|---|---|---|---|---|---|
| 20 | 18,9 | 45 ms | 120 ms | 610 ms | 0 % |
| 50 | 43,2 | 82 ms | 360 ms | 2,3 s | 0 % |
| 100 | 47,2 | 230 ms | 4,7 s | 14 s | **1,8 %** (HTTP 500) |

## Analyse

1. **Un processus sature vers 15 à 20 req/s.** Au-delà de ~20 utilisateurs, le débit plafonne et ce sont les temps de réponse qui augmentent : une médiane de 4,5 s à 100 utilisateurs. Même `/health`, qui ne fait aucun calcul, ralentit. Le goulot n'est donc pas le réseau mais le **CPU d'un seul processus Python** : la prédiction des 100 arbres et l'écriture en base passent par un seul cœur, à cause du GIL.
2. **Passer à 4 workers multiplie le débit utile par environ 2,5** : 43 req/s avec un p95 de 360 ms à 50 utilisateurs, contre 1,8 s avec un seul processus. Le gain n'est pas de ×4, car Locust et MLflow tournent sur la même machine et prennent du CPU.
3. **SQLite casse sous la concurrence.** À 100 utilisateurs avec 4 workers, 1,8 % des requêtes échouent en HTTP 500. Les logs de l'API montrent `sqlite3.OperationalError: database is locked` : plusieurs processus écrivent en même temps dans le même fichier. Avec un seul processus, ce problème n'apparaît pas, mais le débit est limité.
4. **Le contrat de l'API est respecté sous charge** : la version du modèle est présente dans chaque réponse réussie (contrôlé par le script Locust), et chaque prédiction est enregistrée.

## Recommandation de dimensionnement

**Besoin réel estimé.** Le pic observé dans les données est de 977 locations par heure, soit moins de 0,3 location par seconde. Même si chaque location déclenchait une prédiction, un seul processus suffirait largement. La charge à prévoir vient plutôt des consultations simultanées (outils de planification, tableaux de bord) : on retient comme objectif **50 utilisateurs simultanés avec un p95 sous 500 ms**.

**Configuration recommandée :**

| Élément | Recommandation | Justification |
|---|---|---|
| Processus API | **4 workers** (1 par cœur physique) | 43 req/s et p95 de 360 ms à 50 utilisateurs, contre 1,8 s avec un seul processus |
| Base de données | **PostgreSQL**, obligatoire dès qu'il y a plus d'un worker | SQLite renvoie `database is locked` avec plusieurs processus |
| Capacité par worker | environ **10 req/s** en gardant un p95 sous 500 ms | mesuré : 43 req/s pour 4 workers |
| Règle de calcul | workers = débit visé / 10, **+30 % de marge** | Exemple : 100 req/s → 10 workers, donc 3 instances de 4 workers |
| Mise à l'échelle | horizontale (plusieurs conteneurs API derrière un répartiteur) | le service est sans état : modèle en mémoire, données en base |
| Alerte | p95 > 500 ms ou taux d'erreur > 1 % | ce sont les seuils au-delà desquels la qualité de service se dégrade dans les mesures |

**Pistes d'optimisation**, si la charge augmente :
- un endpoint de prédiction par lot (plusieurs heures en une requête) ;
- une écriture en base asynchrone, hors du temps de réponse ;
- un modèle plus léger (moins d'arbres), à valider contre la règle de promotion.

## Limites

- Locust tourne sur la même machine que l'API : il lui prend du CPU, donc les débits mesurés sont **sous-estimés**.
- Paliers de 60 s : les résultats sont bruités, par exemple l'écart entre 30 et 50 utilisateurs avec un seul processus. Des paliers plus longs et répétés affineraient les chiffres.
- Les mesures ont été faites sur un poste de développement Windows, pas sur l'infrastructure cible. Il faudra les refaire dans Docker avec PostgreSQL.
