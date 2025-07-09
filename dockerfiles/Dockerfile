# Utiliser une image officielle Python comme base
FROM python:3.10-slim

# Définir le répertoire de travail à l'intérieur du conteneur
WORKDIR /app

# Installer git et curl
RUN apt-get update && \
    apt-get install -y git curl gcc libffi-dev libssl-dev && \
    rm -rf /var/lib/apt/lists/* 


# Installer les dépendances Python
COPY ./requirements/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt


CMD ["python", "main.py"]