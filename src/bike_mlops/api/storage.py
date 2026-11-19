"""Persistance des prédictions (entrées, version, durée, erreurs) et des retours terrain."""
import os
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


def now() -> datetime:
    return datetime.now(timezone.utc)


class PredictionLog(Base):
    """Une ligne par appel à /predict, réussi ou non (statut "ok" ou "error")."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    model_version: Mapped[str | None] = mapped_column(String(20))
    features: Mapped[dict | None] = mapped_column(JSON)
    prediction: Mapped[float | None] = mapped_column(Float)
    duration_ms: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(10))
    error: Mapped[str | None] = mapped_column(String(500))


class Feedback(Base):
    """Valeur réellement observée, rattachée à une prédiction."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"))
    actual: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


def get_database_url(cfg: dict) -> str:
    """Adresse de la base : la variable d'environnement est prioritaire sur la config."""
    return os.getenv("DATABASE_URL", cfg["api"]["database_url"])


class PredictionStore:
    """Écrit les prédictions et les retours dans la base (SQLite en local, PostgreSQL en Docker)."""

    def __init__(self, url: str) -> None:
        self.engine = create_engine(url)
        Base.metadata.create_all(self.engine)

    def log_prediction(self, **fields) -> int:
        """Enregistre un appel à /predict. Renvoie son identifiant."""
        with Session(self.engine) as session:
            row = PredictionLog(**fields)
            session.add(row)
            session.commit()
            return row.id

    def add_feedback(self, prediction_id: int, actual: float) -> bool:
        """Enregistre la valeur observée. Renvoie False si la prédiction n'existe pas."""
        with Session(self.engine) as session:
            if session.get(PredictionLog, prediction_id) is None:
                return False
            session.add(Feedback(prediction_id=prediction_id, actual=actual))
            session.commit()
            return True
