"""Métriques métier exposées à Prometheus sur /metrics, en plus des métriques HTTP de l'instrumentation."""
from prometheus_client import Counter, Gauge, Histogram

PREDICTIONS = Counter(
    "bike_predictions_total",
    "Nombre d'appels à /predict, par version du modèle et par statut (ok / error)",
    ["model_version", "status"],
)

PREDICTED_VALUE = Histogram(
    "bike_prediction_value",
    "Distribution des valeurs prédites (nombre de locations par heure)",
    buckets=(0, 25, 50, 100, 200, 300, 400, 500, 650, 800, 1000),
)

FEEDBACK_ABS_ERROR = Histogram(
    "bike_feedback_abs_error",
    "Écart absolu entre la prédiction et la valeur observée remontée par /feedback",
    ["model_version"],
    buckets=(5, 10, 25, 50, 75, 100, 150, 200, 300, 500),
)

MODEL_VERSION = Gauge(
    "bike_model_version",
    "Numéro de la version du modèle champion actuellement servie (0 si aucun modèle chargé)",
)
