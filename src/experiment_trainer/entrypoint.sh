#!/bin/bash

# Attente pour laisser le temps au serveur de démarrer
echo "➡️  Attente que MLflow server soit prêt..."
for i in {1..20}; do
  if curl -f http://mlflow-server:5050; then
    echo "✅ MLflow server prêt."
    break
  else
    echo "⏳ MLflow server pas encore prêt, attente..."
    sleep 3
  fi
done

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