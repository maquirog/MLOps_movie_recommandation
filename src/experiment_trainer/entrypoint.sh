#!/bin/bash

echo "➡️  Lancement du serveur MLflow..."
mlflow server \
  --host 0.0.0.0 \
  --port 5050 \
  --backend-store-uri /app/mlruns \
  --default-artifact-root /app/mlruns &

# Attente pour laisser le temps au serveur de démarrer
sleep 5

# Génère le nom d'expérimentation dynamiquement
now=$(date +"%Y-%m-%d_%H-%M-%S")
experiment_name="weekly_experiment_${now}"

# Crée l'expérience weekly
mlflow experiments create --experiment-name "$experiment_name" || true

echo "➡️  Lancement de l'expérience MLflow..."
#python src/experiment_trainer/grid_search_experiment.py
mlflow run src/experiment_trainer \
  --env-manager=local \
  -P experiment_name="$experiment_name"

echo "✅ Expérience terminée. MLflow disponible sur http://localhost:5050"
tail -f /dev/null
