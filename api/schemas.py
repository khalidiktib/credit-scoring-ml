"""
Schémas Pydantic : définissent le contrat de l'API.

CreditApplicationInput utilise des alias pour mapper des noms Python propres
vers les noms de colonnes exacts attendus par le pipeline sklearn (certains
contiennent des tirets, invalides comme identifiants Python).

Les bornes (ge=, le=...) ne sont pas arbitraires : elles reflètent les
limites métier réalistes qu'on a identifiées en EDA (ex: age doit être > 0,
on a vu une ligne à 0 qui était une erreur de saisie).
"""
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class CreditApplicationInput(BaseModel):
    revolving_utilization: float = Field(
        ..., alias="RevolvingUtilizationOfUnsecuredLines", ge=0,
        description="Ratio d'utilisation du crédit renouvelable disponible",
    )
    age: int = Field(..., alias="age", gt=17, le=110, description="Âge de l'emprunteur")
    late_30_59: int = Field(
        ..., alias="NumberOfTime30-59DaysPastDueNotWorse", ge=0, le=20,
        description="Nombre de retards de paiement de 30-59 jours",
    )
    debt_ratio: float = Field(..., alias="DebtRatio", ge=0, description="Ratio dette/revenu")
    monthly_income: Optional[float] = Field(
        None, alias="MonthlyIncome", ge=0, description="Revenu mensuel (optionnel)",
    )
    open_credit_lines: int = Field(
        ..., alias="NumberOfOpenCreditLinesAndLoans", ge=0,
        description="Nombre de lignes de crédit ouvertes",
    )
    late_90: int = Field(
        ..., alias="NumberOfTimes90DaysLate", ge=0, le=20,
        description="Nombre de retards de paiement de 90+ jours",
    )
    real_estate_loans: int = Field(
        ..., alias="NumberRealEstateLoansOrLines", ge=0,
        description="Nombre de prêts immobiliers",
    )
    late_60_89: int = Field(
        ..., alias="NumberOfTime60-89DaysPastDueNotWorse", ge=0, le=20,
        description="Nombre de retards de paiement de 60-89 jours",
    )
    dependents: Optional[int] = Field(
        None, alias="NumberOfDependents", ge=0, description="Nombre de personnes à charge",
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "RevolvingUtilizationOfUnsecuredLines": 0.3,
                "age": 45,
                "NumberOfTime30-59DaysPastDueNotWorse": 0,
                "DebtRatio": 0.25,
                "MonthlyIncome": 5000,
                "NumberOfOpenCreditLinesAndLoans": 6,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 1,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 2,
            }
        },
    )


class CreditScoreOutput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    default_probability: float = Field(..., description="Probabilité de défaut à 2 ans (0-1)")
    risk_level: str = Field(..., description="low / medium / high")
    model_version: str = Field(..., description="Version du modèle utilisé (Registry)")