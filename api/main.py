"""
API FastAPI exposant le modèle de scoring de crédit.

Usage local : uvicorn api.main:app --reload
Puis : http://localhost:8000/docs (documentation interactive auto-générée)
"""
import sys
sys.path.insert(0, ".")

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from api.schemas import CreditApplicationInput, CreditScoreOutput
from api.predictor import CreditScorer

# État partagé de l'app -- alternative aux variables globales, recommandée
# par FastAPI pour stocker des ressources coûteuses (ici : le modèle chargé).
ml_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Charge le modèle UNE FOIS au démarrage du serveur (pas à chaque requête).
    C'est le "lifespan" FastAPI -- le code avant yield tourne au démarrage,
    celui après yield tournerait à l'arrêt (rien à nettoyer ici).
    """
    ml_state["scorer"] = CreditScorer()
    yield
    ml_state.clear()


app = FastAPI(
    title="Credit Scoring API",
    description="Prédit la probabilité de défaut de paiement à 2 ans",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    """
    Endpoint de santé -- standard en production (Docker/Kubernetes s'en
    servent pour savoir si le conteneur est prêt à recevoir du trafic).
    """
    model_loaded = "scorer" in ml_state
    return {"status": "ok" if model_loaded else "model not loaded", "model_loaded": model_loaded}


@app.post("/predict", response_model=CreditScoreOutput)
def predict(application: CreditApplicationInput):
    """
    Reçoit un dossier de crédit, retourne la probabilité de défaut et un
    niveau de risque catégorisé.
    """
    scorer = ml_state.get("scorer")
    if scorer is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé -- réessaie dans un instant.")

    # by_alias=True : reconstitue les noms de colonnes exacts attendus par
    # le pipeline sklearn (ex: 'NumberOfTime30-59DaysPastDueNotWorse')
    input_dict = application.model_dump(by_alias=True)
    result = scorer.predict(input_dict)
    return CreditScoreOutput(**result)