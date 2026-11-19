"""Schémas d'entrée et de sortie de l'API (validation automatique par FastAPI)."""
from pydantic import BaseModel, ConfigDict, Field


class BikeFeatures(BaseModel):
    """Conditions d'une heure donnée. Mêmes plages que le contrôle qualité des lots."""

    model_config = ConfigDict(extra="forbid")  # un champ inconnu est refusé

    season: int = Field(ge=1, le=4, description="1 hiver, 2 printemps, 3 été, 4 automne")
    yr: int = Field(ge=0, le=1, description="0 = 2011, 1 = 2012")
    mnth: int = Field(ge=1, le=12)
    hr: int = Field(ge=0, le=23)
    holiday: int = Field(ge=0, le=1)
    weekday: int = Field(ge=0, le=6)
    workingday: int = Field(ge=0, le=1)
    weathersit: int = Field(ge=1, le=4, description="1 dégagé ... 4 très mauvais temps")
    temp: float = Field(ge=0, le=1, description="température normalisée")
    atemp: float = Field(ge=0, le=1, description="température ressentie normalisée")
    hum: float = Field(ge=0, le=1, description="humidité normalisée")
    windspeed: float = Field(ge=0, le=1, description="vent normalisé")


class PredictionResponse(BaseModel):
    """Réponse : la prédiction et la version du modèle qui l'a produite."""

    model_config = ConfigDict(protected_namespaces=())  # autorise les champs model_*

    prediction: float
    model_name: str
    model_version: str
