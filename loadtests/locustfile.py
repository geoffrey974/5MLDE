"""Scénario de charge : des utilisateurs demandent des prédictions et renvoient parfois la valeur observée."""
import random

from locust import HttpUser, between, task


def random_features() -> dict:
    """Une heure tirée au hasard, toujours dans les plages acceptées par l'API."""
    mnth = random.randint(1, 12)
    weekday = random.randint(0, 6)
    holiday = int(random.random() < 0.03)
    return {
        "season": (mnth % 12) // 3 + 1,
        "yr": random.randint(0, 1),
        "mnth": mnth,
        "hr": random.randint(0, 23),
        "holiday": holiday,
        "weekday": weekday,
        "workingday": int(weekday not in (0, 6) and not holiday),
        "weathersit": random.choices([1, 2, 3, 4], weights=[65, 26, 8, 1])[0],
        "temp": round(random.uniform(0.02, 1), 2),
        "atemp": round(random.uniform(0, 1), 2),
        "hum": round(random.uniform(0, 1), 2),
        "windspeed": round(random.uniform(0, 0.85), 2),
    }


class ApiUser(HttpUser):
    """Utilisateur type : surtout des prédictions, quelques retours terrain et vérifications d'état."""

    wait_time = between(0.5, 1.5)

    def on_start(self) -> None:
        self.prediction_ids: list[int] = []

    @task(10)
    def predict(self) -> None:
        with self.client.post("/predict", json=random_features(), catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return
            body = response.json()
            if "model_version" not in body:
                response.failure("version du modèle absente de la réponse")
                return
            self.prediction_ids.append(body["prediction_id"])

    @task(2)
    def feedback(self) -> None:
        if not self.prediction_ids:
            return
        prediction_id = self.prediction_ids.pop(0)
        self.client.post("/feedback", json={"prediction_id": prediction_id, "actual": random.randint(0, 900)})

    @task(1)
    def health(self) -> None:
        self.client.get("/health")
