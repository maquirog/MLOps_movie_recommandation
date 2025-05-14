#!/bin/bash

echo "➡️  Lancement du serveur MLflow..."
mlflow server \
  --host 0.0.0.0 \
  --port 8080 \
  --backend-store-uri /app/mlruns \
  --default-artifact-root /app/mlruns &

# Attente pour laisser le temps au serveur de démarrer
sleep 5

echo "➡️  Lancement de l'expérience MLflow..."
mlflow run mlflow_experiment \
  --env-manager=local \
  --experiment-name=Movie_Recommandation_Model \
  #-P n_movies_metrics=20

echo "✅ Expérience terminée. MLflow disponible sur http://localhost:8080"
tail -f /dev/null