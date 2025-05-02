# Dockerfile pour le service d'évaluation
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY ./src/models/evaluate /app
COPY ./data /app/data
COPY ./models /app/models

# Installer les dépendances
RUN pip install --no-cache-dir -r /app/requirements.txt

# Commande par défaut pour lancer l'évaluation
CMD ["python", "evaluate.py"]