# Synthèse OPS — Prévision de la demande de vélos

> **Service** : prédiction du nombre de locations de vélos par heure (jeu UCI Bike Sharing, 2011-2012)
> **Déploiement** : Docker Compose, 7 services
> **Dépôt** : https://github.com/geoffrey974/5MLDE

---

## Résumé

Le projet met en production un modèle de prévision de la demande et couvre toute sa chaîne d'exploitation. Les données arrivent par lots mensuels. Chaque lot est contrôlé, puis le modèle est réentraîné, comparé au modèle en service et promu seulement s'il fait mieux. L'API sert toujours le modèle en service, enregistre chaque prédiction et est supervisée en continu.

| Indicateur                             | Valeur                                             |
| -------------------------------------- | -------------------------------------------------- |
| Tests automatisés                      | **40**, tous au vert                               |
| Capacité (4 workers + PostgreSQL)      | **~51 req/s**, **0 %** d'erreur à 100 utilisateurs |
| Latence nominale de `/predict`         | p95 ≈ **240 ms** à 15 utilisateurs                 |
| Application d'une promotion / rollback | **< 40 s**, sans redémarrage                       |
| Incidents simulés et maîtrisés         | **4**                                              |

---

## 1. Périmètre

| Exigence                  | Mise en œuvre                                            |
| ------------------------- | -------------------------------------------------------- |
| Ingestion reproductible   | téléchargement scripté, SHA-256, 24 lots mensuels        |
| Contrôle qualité bloquant | schéma Pandera, un lot invalide arrête la chaîne         |
| Orchestration et alertes  | Prefect : détection toutes les 60 s, retries, alertes    |
| Suivi et versionnage      | MLflow : paramètres, métriques, données, artefacts, tags |
| Promotion et rollback     | règle de mise en service, alias `champion`               |
| Service de prédiction     | FastAPI : validation, version dans chaque réponse        |
| Persistance et feedback   | PostgreSQL : prédictions, durées, erreurs, feedback      |
| Supervision               | Prometheus + Grafana : tableau de bord, 4 alertes        |
| Dimensionnement           | tests de charge Locust, recommandation chiffrée          |

L'arrivée de nouvelles données est simulée en rejouant le jeu historique, un mois à la fois.

---

## 2. Architecture

```
 Lots mensuels ──> Pipeline Prefect ──> contrôle qualité ──> entraînement ──> promotion
                                                                  │
                                                                  v
                                                   MLflow (suivi + registre)
                                                                  │ alias "champion"
                                                                  v
 Client ──> API FastAPI (4 workers) ──> PostgreSQL (prédictions, erreurs, feedback)
                │
                v
           Prometheus (alertes) ──> Grafana (tableau de bord)
```

| Service               | Technologie       | Rôle                                       |
| --------------------- | ----------------- | ------------------------------------------ |
| `pipeline`, `prefect` | Prefect 3         | orchestration, déclenchement automatique   |
| `mlflow`              | MLflow 3          | suivi des expériences, registre de modèles |
| `api`                 | FastAPI + Uvicorn | prédictions, feedback, métriques           |
| `postgres`            | PostgreSQL 16     | bases `mlflow` et `predictions`            |
| `prometheus`          | Prometheus 3      | collecte des métriques, alertes            |
| `grafana`             | Grafana 12        | tableau de bord                            |

---

## 3. Traçabilité

Pour n'importe quelle prédiction, on remonte toute la chaîne jusqu'au fichier source :

| Niveau       | Emplacement         | Contenu                                              |
| ------------ | ------------------- | ---------------------------------------------------- |
| Prédiction   | table `predictions` | entrées, sortie, version, durée, statut, erreur      |
| Version      | tags du registre    | période, lot d'évaluation, RMSE, empreinte, décision |
| Entraînement | run MLflow          | hyperparamètres, métriques, features, modèle         |
| Données      | `metadata.json`     | URL source, date, SHA-256 du fichier brut            |

Tags enregistrés sur chaque version : `train_until`, `eval_batch`, `rmse`, `data_sha256`, `validation_status`, `promotion_decision` et `promotion_reason`.

**Reproductibilité vérifiée** :
- deux entraînements sur les mêmes lots donnent la même empreinte `data_sha256` et les mêmes métriques (`random_state` fixé) ;
- le même modèle, entraîné en local puis dans Docker, donne exactement la même prédiction (483,08 pour la requête de référence).

---

## 4. Mise en service et retour arrière

Une nouvelle version devient **champion** seulement si elle respecte, dans l'ordre :

1. **Qualité minimale** : RMSE inférieure à `max_rmse = 100`.
2. **Fraîcheur** : données d'entraînement plus récentes que celles du champion. Sinon, le champion a pu voir le mois d'évaluation, et la comparaison serait faussée.
3. **Performance** : RMSE plus basse que celle du champion, **sur le même lot d'évaluation**.

Chaque décision est enregistrée sur la version (`promotion_decision`, `promotion_reason`).

**Cas observés** :
- version moins performante → rejetée (règle 3) ;
- version aux données plus anciennes → rejetée (règle 2).

**Retour arrière** : l'alias `champion` est replacé sur la version précédente (tag `previous_champion`) ou sur une version choisie. L'API vérifie le registre toutes les 30 s : promotion et rollback s'appliquent **sans redémarrage**. Mesuré : v1 → v2 → v1 en moins de 40 s à chaque fois.

---

## 5. Gestion des incidents

| Incident        | Détection         | Réaction                                  | Remédiation         |
| --------------- | ----------------- | ----------------------------------------- | ------------------- |
| Lot corrompu    | Pandera           | run `Failed`, pas d'entraînement, alerte  | remplacer le lot    |
| MLflow coupé    | connexion refusée | API maintenue, pipeline en échec + alerte | redémarrer MLflow   |
| API sans modèle | chargement échoué | `degraded`, 503, alerte                   | rechargement auto   |
| API arrêtée     | `up == 0`         | alerte, Grafana « HORS LIGNE »            | redémarrage auto    |
| Saturation      | p95 > 500 ms      | alerte latence ou erreurs                 | ajouter des workers |
| Entrée invalide | Pydantic          | 422, erreur enregistrée                   | aucune              |

Détails :
- **Lot corrompu** : le champion et l'API ne sont pas touchés. Une fois le lot remplacé, le pipeline reprend tout seul au cycle suivant.
- **MLflow coupé** : l'API continue de servir le modèle déjà chargé en mémoire. Le pipeline réessaie plusieurs fois avant d'échouer.
- **API sans modèle** : si MLflow est absent au démarrage, l'API reste joignable (`/health` = `degraded`) et charge le modèle dès le retour de MLflow.
- **Alertes du pipeline** : elles sont tracées dans les logs, dans `logs/alerts.jsonl` et sur un webhook optionnel (Discord, Slack).

**Problème rencontré et corrigé** : sur une base vide, le premier démarrage de MLflow pouvait échouer, car ses workers créaient le schéma en même temps. Correctif : `mlflow db upgrade` est exécuté avant le serveur.

---

## 6. Dimensionnement

Protocole : Locust, paliers de 60 s, 10 prédictions pour 2 feedbacks et 1 `/health`. Détail : `loadtests/RESULTATS.md`.

| Configuration              | Débit max     | p95 (100 util.) | Erreurs (100 util.) |
| -------------------------- | ------------- | --------------- | ------------------- |
| 1 processus + SQLite       | 15 à 20 req/s | 9,4 s           | 0 %                 |
| 4 workers + SQLite         | ~47 req/s     | 4,7 s           | **1,8 %**           |
| **4 workers + PostgreSQL** | **~51 req/s** | **1,7 s**       | **0 %**             |

**Constats** :
- un seul processus Python sature vite : le calcul du RandomForest est limité à un cœur (GIL) ;
- plusieurs workers multiplient le débit, mais SQLite ne supporte pas les écritures concurrentes (`database is locked`) ;
- PostgreSQL supprime les erreurs.

**Recommandation** :
- **4 workers par instance** (1 par cœur), avec **PostgreSQL** ;
- capacité de référence : **~10 req/s par worker** en gardant un p95 sous 500 ms ;
- objectif : **50 utilisateurs simultanés** par instance ; au-delà, ajouter des instances (workers = débit visé ÷ 10, + 30 % de marge).

Le besoin métier reste modeste : le pic historique est de 977 locations par heure, soit moins de 0,3 req/s. Le dimensionnement vise les pics de consultation, pas le volume de locations.

---

## 7. Supervision

| Axe     | Indicateurs                                                 |
| ------- | ----------------------------------------------------------- |
| Service | disponibilité, débit, latence p50/p95, taux d'erreur        |
| Modèle  | version servie, prédictions par version, erreur du feedback |
| Alertes | API indisponible, aucun modèle, p95 > 500 ms, 5xx > 1 %     |

Les seuils d'alerte viennent directement des mesures de dimensionnement. Avec 4 workers, les métriques sont agrégées sur tous les processus. Vérifié : 2 342 prédictions comptées par Prometheus, et 2 342 lignes en base.

**Dérive** : un modèle entraîné uniquement sur 2011 n'obtient qu'un R² de 0,36 sur janvier 2012, car la demande augmente d'une année sur l'autre. Le réentraînement mensuel, la règle de fraîcheur et le suivi de l'erreur par le feedback répondent à ce risque.

---

## 8. Choix techniques

- **Sérialisation sûre** : les modèles sont enregistrés avec skops plutôt que pickle. Seul le type `sklearn.tree._tree.Tree` est autorisé au chargement, ce qui évite d'exécuter du code arbitraire.
- **Secrets** : aucun secret dans le dépôt. Les mots de passe passent par `.env` (ignoré par Git), avec `.env.example` comme modèle.
- **Conteneurs** : l'image applicative tourne avec un utilisateur non root, sauf MLflow, qui doit écrire dans son volume d'artefacts.
- **Données** : non versionnées, retéléchargées et vérifiées par empreinte.
- **Code** : modules dans `src/`, configuration dans `configs/config.yaml`, dépendances figées, 40 tests, une branche et une pull request par étape (`feature/*` → `develop` → `main`).

---

## 9. Limites et pistes d'amélioration

| Limite                                     | Piste d'amélioration               |
| ------------------------------------------ | ---------------------------------- |
| Un seul `requirements.txt`                 | séparer exécution et développement |
| Pas d'Alertmanager                         | notifier les alertes Prometheus    |
| API sans authentification ni HTTPS         | clé d'API ou OAuth2, proxy TLS     |
| Dérive suivie par le feedback seulement    | test statistique (PSI)             |
| Fenêtre d'entraînement croissante          | fenêtre glissante, à évaluer       |
| Charge testée sur un poste de dev          | refaire sur l'infrastructure cible |
| Avertissements des bibliothèques           | suivre les montées de version      |
| 17 379 lignes au lieu des 17 389 annoncées | écart documenté                    |
