# Utilise une image légère Python
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires dans l'image
COPY ./src/data /app

# Installer les dépendances Python
RUN pip install --no-cache-dir -r /app/requirements.txt

# Définir une commande par défaut (modifiable dans docker-compose.yml)
CMD ["python", "import_raw_data.py"]