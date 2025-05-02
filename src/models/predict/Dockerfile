# Dockerfile pour le service de prédiction
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY ./src/models/predict /app
COPY ./data /app/data
COPY ./models /app/models

# Installer les dépendances
RUN pip install --no-cache-dir -r /app/requirements.txt

# Commande par défaut pour lancer les prédictions
CMD ["python", "predict.py"]