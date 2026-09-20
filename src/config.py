"""
Configuration centralisée du projet : chemins et constantes.

Pourquoi centraliser ici plutôt que coder en dur dans chaque script ?
- Un seul endroit à modifier si l'arborescence change
- Facile à surcharger avec des variables d'environnement une fois dockerisé
  (ex: DATA_DIR devient configurable via docker-compose.yml)
"""
import os
from pathlib import Path

# Racine du projet (2 niveaux au-dessus de ce fichier : src/config.py -> racine)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Chemins des données
DATA_RAW_DIR = Path(os.getenv("DATA_RAW_DIR", PROJECT_ROOT / "data" / "raw"))
DATA_PROCESSED_DIR = Path(os.getenv("DATA_PROCESSED_DIR", PROJECT_ROOT / "data" / "processed"))
RAW_DATA_FILE = DATA_RAW_DIR / "cs-training.csv"

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file:{PROJECT_ROOT / 'mlruns'}")
MLFLOW_EXPERIMENT_NAME = "credit-scoring"

# Modèles
MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models"))
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Target
TARGET_COLUMN = "SeriousDlqin2yrs"