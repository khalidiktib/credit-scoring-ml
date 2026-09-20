"""
Charge le pipeline complet (preprocessing + modèle) depuis le MLflow Model
Registry, et expose une fonction de prédiction simple.

On charge par nom + stage ("models:/credit-scoring-model/Staging"), pas par
run_id en dur -- ça veut dire que promouvoir une nouvelle version dans le
Registry (Staging -> Production, ou une nouvelle version en Staging) ne
nécessite AUCUN changement de code ici, juste un redémarrage de l'API.
"""
import pandas as pd
import mlflow

from src.config import MLFLOW_TRACKING_URI

REGISTERED_MODEL_NAME = "credit-scoring-model"
MODEL_STAGE = "Staging"

# Seuils métier pour catégoriser le risque -- à ajuster selon la politique
# de la banque. Ici choisis arbitrairement pour l'exemple.
RISK_THRESHOLDS = {"low": 0.10, "medium": 0.30}  # au-delà de 0.30 -> "high"


class CreditScorer:
    """Encapsule le modèle chargé pour éviter de le recharger à chaque appel."""

    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
        # mlflow.sklearn.load_model (pas pyfunc) -- on récupère l'objet Pipeline
        # sklearn natif pour pouvoir appeler predict_proba(), pas juste predict().
        self.pipeline = mlflow.sklearn.load_model(model_uri)
        self.model_version = self._resolve_version()

    def _resolve_version(self) -> str:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=[MODEL_STAGE])
        return str(versions[0].version) if versions else "unknown"

    def predict(self, input_dict: dict) -> dict:
        """
        input_dict : colonnes exactes attendues par le pipeline (noms originaux,
        ex: 'NumberOfTime30-59DaysPastDueNotWorse'), obtenues via
        CreditApplicationInput.model_dump(by_alias=True) côté API.
        """
        X = pd.DataFrame([input_dict])
        proba = float(self.pipeline.predict_proba(X)[0, 1])

        if proba < RISK_THRESHOLDS["low"]:
            risk_level = "low"
        elif proba < RISK_THRESHOLDS["medium"]:
            risk_level = "medium"
        else:
            risk_level = "high"

        return {
            "default_probability": round(proba, 4),
            "risk_level": risk_level,
            "model_version": self.model_version,
        }