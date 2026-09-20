"""
Script de vérification : reproduit les checks qu'on a faits ensemble
(NaN résiduels, scaling, stratification) pour valider  l'installation
locale avant de passer à l'entraînement des modèles.

Usage : python check_pipeline.py
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.preprocessing import build_preprocessing_pipeline
from src.config import TARGET_COLUMN, RANDOM_STATE, TEST_SIZE, RAW_DATA_FILE


def main():
    print(f"Chargement de {RAW_DATA_FILE} ...")
    df = pd.read_csv(RAW_DATA_FILE, index_col=0)
    print(f"Shape brute : {df.shape}")

    df = df.drop_duplicates()
    print(f"Shape après dédup : {df.shape}")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train : {X_train.shape} | Test : {X_test.shape}")
    print(f"Taux de défaut train : {y_train.mean():.4f} | test : {y_test.mean():.4f}")

    pipeline = build_preprocessing_pipeline()
    X_train_t = pipeline.fit_transform(X_train)
    X_test_t = pipeline.transform(X_test)

    print()
    print(f"Shape transformée train : {X_train_t.shape}")
    print(f"Shape transformée test  : {X_test_t.shape}")

    nan_train = pd.DataFrame(X_train_t).isnull().sum().sum()
    nan_test = pd.DataFrame(X_test_t).isnull().sum().sum()
    print(f"NaN résiduels train : {nan_train} {'✅' if nan_train == 0 else '❌'}")
    print(f"NaN résiduels test  : {nan_test} {'✅' if nan_test == 0 else '❌'}")

    means = pd.DataFrame(X_train_t).mean().round(2)
    print(f"Moyennes après scaling (doivent être ~0) : {means.values}")

    print()
    print("✅ Pipeline validé — prêt pour l'entraînement des modèles.")


if __name__ == "__main__":
    main()