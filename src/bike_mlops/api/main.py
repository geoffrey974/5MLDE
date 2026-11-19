"""API de prédiction : sert le modèle champion du registre MLflow et enregistre chaque appel."""
import logging
import time
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from mlflow import MlflowClient

from bike_mlops.api.schemas import BikeFeatures, FeedbackRequest, PredictionResponse
from bike_mlops.api.storage import PredictionStore, get_database_url
from bike_mlops.config import get_tracking_uri, load_config
from bike_mlops.models.registry import CHAMPION

logger = logging.getLogger(__name__)


class ModelService:
    """Garde en mémoire le modèle champion et son numéro de version."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.name = cfg["model"]["name"]
        self.model = None
        self.version: str | None = None

    def load(self) -> str:
        """Charge la version portant l'alias champion. Renvoie son numéro."""
        mlflow.set_tracking_uri(get_tracking_uri(self.cfg))
        champion = MlflowClient().get_model_version_by_alias(self.name, CHAMPION)
        self.model = mlflow.sklearn.load_model(f"models:/{self.name}/{champion.version}")
        self.version = str(champion.version)
        logger.info("Modèle %s v%s chargé", self.name, self.version)
        return self.version


cfg = load_config()
service = ModelService(cfg)
store = PredictionStore(get_database_url(cfg))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Au démarrage : charge le champion. Si MLflow est indisponible, l'API démarre quand même (503)."""
    try:
        service.load()
    except Exception as exc:
        logger.error("Modèle non chargé au démarrage : %s", exc)
    yield


app = FastAPI(title="Bike demand API", version="1.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_input(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Entrée invalide : réponse 422, et l'erreur est enregistrée comme les prédictions."""
    if request.url.path == "/predict":
        first = exc.errors()[0]
        store.log_prediction(
            model_version=service.version,
            features=exc.body if isinstance(exc.body, dict) else None,
            duration_ms=0.0,
            status="error",
            error=f"422 entrée invalide : {first['loc'][-1]} - {first['msg']}"[:500],
        )
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Erreur imprévue : réponse 500 propre, détail dans les logs uniquement."""
    logger.exception("Erreur sur %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Erreur interne du serveur"})


@app.get("/health")
def health() -> dict:
    """État du service : 'ok' si un modèle est chargé, 'degraded' sinon."""
    return {
        "status": "ok" if service.model is not None else "degraded",
        "model_name": service.name,
        "model_version": service.version,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: BikeFeatures) -> PredictionResponse:
    """Prédit le nombre de locations pour une heure donnée (jamais négatif). Chaque appel est enregistré."""
    start = time.perf_counter()
    try:
        if service.model is None:
            raise HTTPException(status_code=503, detail="Aucun modèle chargé")
        X = pd.DataFrame([features.model_dump()])
        prediction = round(max(0.0, float(service.model.predict(X)[0])), 2)
    except Exception as exc:
        detail = exc.detail if isinstance(exc, HTTPException) else repr(exc)
        store.log_prediction(
            model_version=service.version,
            features=features.model_dump(),
            duration_ms=(time.perf_counter() - start) * 1000,
            status="error",
            error=str(detail)[:500],
        )
        raise

    prediction_id = store.log_prediction(
        model_version=service.version,
        features=features.model_dump(),
        prediction=prediction,
        duration_ms=(time.perf_counter() - start) * 1000,
        status="ok",
    )
    return PredictionResponse(
        prediction_id=prediction_id, prediction=prediction, model_name=service.name, model_version=service.version
    )


@app.post("/feedback", status_code=201)
def feedback(body: FeedbackRequest) -> dict:
    """Enregistre la valeur réellement observée pour une prédiction passée."""
    if not store.add_feedback(body.prediction_id, body.actual):
        raise HTTPException(status_code=404, detail=f"Prédiction {body.prediction_id} inconnue")
    return {"status": "enregistré", "prediction_id": body.prediction_id}


@app.post("/reload")
def reload() -> dict:
    """Recharge le champion (après une promotion ou un retour arrière)."""
    try:
        version = service.load()
    except Exception as exc:
        logger.error("Rechargement impossible : %s", exc)
        raise HTTPException(status_code=503, detail="Modèle champion indisponible") from exc
    return {"model_name": service.name, "model_version": version}
