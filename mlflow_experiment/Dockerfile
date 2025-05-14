FROM python:3.8

# Installer les dépendances
RUN pip install --upgrade pip && pip install \
    pandas==2.0.3 \
    numpy==1.24.4 \
    scikit-learn==1.3.2 \
    mlflow==2.17.2

WORKDIR /app

# Copier uniquement les dossiers nécessaires
COPY src/ /app/src/
COPY data/ /app/data/
COPY mlflow_experiment/ /app/mlflow_experiment/

# Déplacer et rendre exécutable le script d'entrée
RUN cp /app/mlflow_experiment/entrypoint.sh /entrypoint.sh && chmod +x /entrypoint.sh

# Pour que les imports internes fonctionnent
ENV PYTHONPATH=/app

ENTRYPOINT ["/entrypoint.sh"]