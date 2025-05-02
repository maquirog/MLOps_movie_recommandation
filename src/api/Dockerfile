# Utiliser une image officielle Python comme base
FROM python:3.10-slim

# Définir le répertoire de travail à l'intérieur du conteneur
WORKDIR /app

# Copier les fichiers nécessaires dans le conteneur
COPY ./src /app/src
COPY ./requirements.txt /app/
COPY ./models /app/models
COPY ./data /app/data

# Installer les dépendances
RUN pip install --no-cache-dir -r /app/requirements.txt

# Exposer le port sur lequel l'API sera accessible
EXPOSE 8000

# Commande pour exécuter l'API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]