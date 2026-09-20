"""
Cherche le meilleur run de l'expérience (par PR-AUC) et l'enregistre dans
le Model Registry MLflow sous un nom stable ("credit-scoring-model"),
avec transition vers le stage "Staging".

Pourquoi un script séparé de train.py ?
En pratique, tu ne veux pas enregistrer automatiquement CHAQUE run entraîné
dans le registry -- seulement celui que tu as décidé de promouvoir après
avoir comparé les résultats (potentiellement après plusieurs expériences,
plusieurs jours d'itération). Séparer les deux évite qu'un run raté finisse
en Staging par accident.

Usage : python -m src.models.register_best_model
"""
import sys
sys.path.insert(0, ".")

import mlflow
from mlflow.tracking import MlflowClient

from src.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

REGISTERED_MODEL_NAME = "credit-scoring-model"
SELECTION_METRIC = "metrics.pr_auc"


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"Expérience '{MLFLOW_EXPERIMENT_NAME}' introuvable -- lance d'abord "
            f"'python -m src.models.train'."
        )

    # Récupère tous les runs de l'expérience, triés par PR-AUC décroissant
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"{SELECTION_METRIC} DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("Aucun run trouvé dans cette expérience.")

    best_run = runs[0]
    run_id = best_run.info.run_id
    model_name = best_run.data.params.get("model_type", "unknown")
    pr_auc = best_run.data.metrics.get("pr_auc")

    print(f"Meilleur run : {model_name} (run_id={run_id}, PR-AUC={pr_auc:.4f})")

    # Enregistre le modèle de ce run dans le Model Registry
    model_uri = f"runs:/{run_id}/model"
    registered_model = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)

    print(f"Enregistré comme '{REGISTERED_MODEL_NAME}', version {registered_model.version}")

    # Ajoute un tag pour tracer l'algo d'origine (utile une fois plusieurs
    # versions accumulées dans le registry)
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=registered_model.version,
        key="algorithm",
        value=model_name,
    )

    # Transition vers "Staging" : signale "candidat à la production, à valider"
    # avant de passer manuellement à "Production" une fois testé.
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=registered_model.version,
        stage="Staging",
    )
    print(f"Version {registered_model.version} passée en stage 'Staging'.")
    print(f"\nOuvre 'mlflow ui' > onglet 'Models' pour voir le résultat.")


if __name__ == "__main__":
    main()