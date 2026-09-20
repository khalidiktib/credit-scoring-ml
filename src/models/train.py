"""
Entraîne 3 modèles (Logistic Regression, Random Forest, XGBoost), logge
chaque run dans MLflow (paramètres, métriques, modèle sérialisé), et
affiche un résumé pour comparer facilement.

Usage : python -m src.models.train
Puis : mlflow ui   (dans le dossier du projet, pour explorer les runs)
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

from src.data.preprocessing import build_preprocessing_pipeline
from src.config import (
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    RAW_DATA_FILE,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
)


def load_data():
    df = pd.read_csv(RAW_DATA_FILE, index_col=0).drop_duplicates()
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def get_models(y_train):
    """
    Définit les 3 modèles avec gestion du déséquilibre de classe.
    scale_pos_weight pour XGBoost = ratio négatifs/positifs, l'équivalent
    de class_weight='balanced' mais avec sa propre syntaxe.
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos

    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200, scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1,
        ),
    }


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),  # métrique de sélection
    }


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_data()
    models = get_models(y_train)

    results = {}
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline(steps=[
                ("preprocess", build_preprocessing_pipeline()),
                ("classifier", model),
            ])
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            metrics = compute_metrics(y_test, y_pred, y_proba)

            # Log des hyperparamètres du classifieur (pas du pipeline entier)
            mlflow.log_params(model.get_params())
            mlflow.log_param("model_type", name)
            mlflow.log_metrics(metrics)

            # Log du pipeline COMPLET (preprocessing + modèle) -- réutilisable tel quel
            input_example = X_train.head(3).astype("float64")
            mlflow.sklearn.log_model(
                pipeline, artifact_path="model", input_example=input_example
            )

            results[name] = metrics
            print(f"\n{name} :")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

    print("\n=== Résumé (trié par PR-AUC, métrique de sélection) ===")
    ranked = sorted(results.items(), key=lambda x: x[1]["pr_auc"], reverse=True)
    for name, m in ranked:
        print(f"{name:25s} PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}  Recall={m['recall']:.4f}")

    print(f"\n🏆 Meilleur modèle : {ranked[0][0]}")
    print("\nLance 'mlflow ui' dans ce dossier pour explorer les runs en détail.")


if __name__ == "__main__":
    main()
