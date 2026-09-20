"""
Transformers custom basés sur les anomalies identifiées en EDA :
- Codes sentinelles 96/98 dans les colonnes de retard de paiement
- age = 0 (erreur de saisie)
- Outliers extrêmes sur RevolvingUtilization et DebtRatio

Chaque transformer suit l'API sklearn (fit/transform) pour s'intégrer
dans un Pipeline et éviter la fuite de données train/test.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

PAST_DUE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]


class SentinelHandler(BaseEstimator, TransformerMixin):
    """
    Remplace les codes sentinelles (>=96) dans les colonnes de retard de
    paiement par NaN (pour imputation ultérieure), et ajoute un flag
    binaire indiquant qu'une ligne était concernée.

    Pourquoi un flag séparé plutôt qu'une simple imputation silencieuse ?
    Le fait qu'une ligne ait un code sentinelle est potentiellement
    informatif en soi (données manquantes "non au hasard" côté source) :
    on laisse le modèle décider si ce flag a du pouvoir prédictif,
    plutôt que de supprimer l'information en imputant sans trace.
    """

    def __init__(self, cols=None, threshold=96):
        self.cols = cols or PAST_DUE_COLS
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        sentinel_mask = (X[self.cols] >= self.threshold).any(axis=1)
        X["HasSentinelDelinquency"] = sentinel_mask.astype(int)
        for col in self.cols:
            X.loc[X[col] >= self.threshold, col] = np.nan
        return X


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clippe (winsorize) les colonnes à distribution extrême plutôt que de
    supprimer les lignes -- on garde le signal "cas extrême" sans laisser
    une valeur aberrante dominer l'échelle après normalisation.

    Les bornes sont apprises sur le train uniquement (fit), pas recalculées
    sur le test -- c'est ce qui évite la fuite de données ici.
    """

    def __init__(self, cols, lower_quantile=0.001, upper_quantile=0.999):
        self.cols = cols
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        self.bounds_ = {
            col: (
                X[col].quantile(self.lower_quantile),
                X[col].quantile(self.upper_quantile),
            )
            for col in self.cols
        }
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            low, high = self.bounds_[col]
            X[col] = X[col].clip(lower=low, upper=high)
        return X


class AgeFixer(BaseEstimator, TransformerMixin):
    """Remplace age=0 (erreur de saisie) par la médiane apprise sur le train."""

    def fit(self, X, y=None):
        self.median_age_ = X.loc[X["age"] > 0, "age"].median()
        return self

    def transform(self, X):
        X = X.copy()
        X.loc[X["age"] == 0, "age"] = self.median_age_
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Ajoute des features métier interprétables :
    - TotalPastDue : somme des 3 types de retard (signal de risque agrégé)
    - IncomePerDependent : revenu ajusté par personne à charge
    - DebtToIncomeInteraction : combine deux signaux de solvabilité
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["TotalPastDue"] = X[PAST_DUE_COLS].sum(axis=1, skipna=True)
        X["IncomePerDependent"] = X["MonthlyIncome"] / (X["NumberOfDependents"].fillna(0) + 1)
        X["DebtToIncomeInteraction"] = X["DebtRatio"] * X["RevolvingUtilizationOfUnsecuredLines"]
        return X