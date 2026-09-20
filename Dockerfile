# Image de base légère -- "slim" contient juste Python, pas les outils de
# build inutiles à l'exécution (économise ~700 Mo vs l'image python complète).
FROM python:3.11-slim

WORKDIR /app

# Copier requirements.txt AVANT le reste du code : Docker met en cache
# chaque instruction. Si seul ton code change (pas les dépendances), Docker
# réutilise le cache de "pip install" au lieu de tout réinstaller --
# rebuild en secondes au lieu de minutes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier uniquement ce dont l'API a besoin pour tourner -- pas les notebooks,
# pas data/raw (licence Kaggle + inutile en prod), pas tests/.
COPY api/ ./api/
COPY src/ ./src/

# Variable d'environnement lue par src/config.py -- pointe vers le dossier
# mlruns qui sera monté en volume au lancement (voir docker-compose.yml).
ENV MLFLOW_TRACKING_URI=file:/app/mlruns

EXPOSE 8000

# Healthcheck : Docker (et Kubernetes plus tard) peuvent vérifier que le
# conteneur répond vraiment, pas juste qu'il est démarré.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Pas de --reload en prod : ce flag surveille les fichiers pour recharger le
# serveur, un coût inutile (et un risque) une fois en conteneur figé.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]