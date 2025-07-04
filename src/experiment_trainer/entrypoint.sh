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

# Crée l'expérience weekly
mlflow experiments create --experiment-name "$experiment_name" || true


# Prépare l’argument hyperparams s’il est fourni
if [ -z "$2" ]; then
  hyperparams_arg=""
  echo "➡️  Lancement de l'expérience MLflow avec hyperparams prédifinis localement"
else
  hyperparams_arg="--hyperparams_dict '$2'"
  echo "➡️  Lancement de l'expérience MLflow avec hyperparams : $hyperparams_arg"
fi

echo "➡️  Lancement de l'expérience MLflow..."
python  -m src.experiment_trainer.grid_search_experiment --experiment_name "$experiment_name" $hyperparams_arg


echo "✅ Expérience terminée. MLflow disponible sur http://localhost:5050"
