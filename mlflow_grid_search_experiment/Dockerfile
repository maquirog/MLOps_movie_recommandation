FROM python:3.8

WORKDIR /app

# Copier le code source
COPY .. /app

# Installer uniquement les dépendances nécessaires avec les versions spécifiques
RUN pip install --upgrade pip \
    && pip install \
        pandas==2.0.3 \
        numpy==1.24.4 \
        scikit-learn==1.3.2 \
        mlflow==2.17.2

ENV PYTHONPATH=/app

ENTRYPOINT ["mlflow", "run", "mlflow_grid_search_experiment", "--env-manager=local", "--experiment-name=Movie_Recommandation_Model", "--run-name=first_run"]
CMD []