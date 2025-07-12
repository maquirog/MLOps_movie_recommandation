#!/bin/bash

# Arguments : $1 = experiment_name (optionnel), $2 = hyperparams JSON string (optionnel)

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

# Gestion du nom de l'expérience
if [ -z "$1" ]; then
  now=$(date +"%Y-%m-%d_%H-%M-%S")
  experiment_name="weekly_experiment_${now}"
else
  experiment_name="$1"
fi
echo "ℹ️  Nom de l'expérience : $experiment_name"


# Hyperparams ?
if [ -z "$2" ]; then
  echo "➡️  Lancement de l'expérience MLflow avec hyperparams prédéfinis localement"
  python -m src.models.grid_search_experiment --experiment_name "$experiment_name"
else
  echo "➡️  Lancement de l'expérience MLflow avec hyperparams"
  python -m src.models.grid_search_experiment --experiment_name "$experiment_name" --hyperparams_dict="$2"
fi


echo "✅ Expérience terminée. MLflow disponible sur http://localhost:5050"
