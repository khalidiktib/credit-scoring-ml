"""
Pipeline de preprocessing : assemble les transformers custom (src/features/engineering.py)
avec l'imputation/scaling standard de sklearn.

Design : deux étages.
1. Un Pipeline de transformers "métier" qui opèrent sur le DataFrame entier
   (sentinelles, outliers, feature engineering) -- car ils ont besoin de voir
   plusieurs colonnes à la fois pour créer de nouvelles features.
2. Un ColumnTransformer final qui impute + scale chaque colonne numérique --
   c'est l'étape générique qu'on applique une fois toutes les colonnes définies.

Séparer les deux évite un ColumnTransformer monolithique difficile à lire,
et rend chaque étape testable indépendamment.
"""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.features.engineering import (
    SentinelHandler,
    OutlierClipper,
    AgeFixer,
    FeatureEngineer,
    PAST_DUE_COLS,
)

OUTLIER_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
    "DebtToIncomeInteraction",
]

# Colonnes finales après feature engineering, transmises au ColumnTransformer
NUMERIC_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
    "HasSentinelDelinquency",
    "TotalPastDue",
    "IncomePerDependent",
    "DebtToIncomeInteraction",
]


def build_feature_pipeline() -> Pipeline:
    """Étape 1 : nettoyage + feature engineering métier (opère sur DataFrame)."""
    return Pipeline(steps=[
        ("age_fixer", AgeFixer()),
        ("sentinel_handler", SentinelHandler(cols=PAST_DUE_COLS)),
        ("feature_engineer", FeatureEngineer()),
        ("outlier_clipper", OutlierClipper(cols=OUTLIER_COLS)),
    ])


def build_preprocessing_pipeline() -> Pipeline:
    """
    Pipeline complet : feature engineering métier + imputation/scaling générique.
    Retourne un objet sklearn standard, réutilisable tel quel dans l'API FastAPI.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    column_transformer = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
    ])

    return Pipeline(steps=[
        ("features", build_feature_pipeline()),
        ("preprocess", column_transformer),
    ])