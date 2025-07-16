#!/bin/bash

set -e  # stop on error

echo "🔧 Initialisation du projet MLOps..."

# Étape 1: Vérification de .env.local
if [ ! -f .env.local ]; then
    echo "❌ Fichier .env.local introuvable. Crée-le avant de lancer ce script."
    exit 1
fi

echo "✅ Fichier .env.local trouvé."

# Étape 2: Création du réseau Docker s'il n'existe pas
NETWORK_NAME="mlops-net"
if ! docker network ls --format '{{.Name}}' | grep -q "^${NETWORK_NAME}$"; then
    echo "🔗 Création du réseau Docker ${NETWORK_NAME}..."
    docker network create ${NETWORK_NAME}
else
    echo "✅ Réseau ${NETWORK_NAME} déjà existant."
fi

# Étape 3: Création des images
echo "🔩 Construction des images..."
docker-compose build import_raw_data

# Étape 4: Lancement des services nécessaires
echo "🚀 Lancement des services de base..."
docker-compose up -d \
    import_raw_data \
    api \
    mysql \
    mlflow-server \
    evidently \
    prometheus \
    grafana

# Étape 4.1: Attente de la fin du container import_raw_data
echo "⏳ Attente de la fin du container 'import_raw_data'..."
CONTAINER_ID=$(docker-compose ps -q import_raw_data)
MAX_RETRIES=30       
WAIT_SECONDS=10


for i in $(seq 1 $MAX_RETRIES); do
    STATUS=$(docker inspect -f '{{.State.Status}}' "$CONTAINER_ID")
    EXIT_CODE=$(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_ID")

        if [ "$STATUS" == "exited" ]; then
          if [ "$EXIT_CODE" == "0" ]; then
              echo "✅ Container 'import_raw_data' terminé avec succès (exit 0)"
              break
          else
              echo "❌ Container 'import_raw_data' terminé avec une erreur (exit $EXIT_CODE)"
              echo "🪵 Logs du container :"
              docker logs "$CONTAINER_ID"
              exit 1
          fi
        else
        echo "⏳ Tentative $i/$MAX_RETRIES : toujours en cours... (status: $STATUS)"
        sleep $WAIT_SECONDS
    fi
done

# Si après la boucle on n'est toujours pas sortis, c’est un échec
FINAL_STATUS=$(docker inspect -f '{{.State.Status}}' "$CONTAINER_ID")
if [ "$FINAL_STATUS" != "exited" ]; then
    echo "❌ Timeout : le container 'import_raw_data' ne s'est pas terminé après $((MAX_RETRIES * WAIT_SECONDS)) secondes."
    exit 1
fi

# Étape 4.2: Check des services
check_service() {
  local name=$1
  local url=$2
  local max_retries=12
  local wait_seconds=5

  echo -n "🔍 Vérification $name..."

  for i in $(seq 1 $max_retries); do
    if curl --silent --fail "$url" > /dev/null; then
      echo " ✅ OK ($url)"
      return 0
    else
      echo -n "."
      sleep $wait_seconds
    fi
  done

  echo " ❌ Échec : $name ne répond pas sur $url"
  return 1
}
check_service "API (FastAPI)"       "http://localhost:8000/health"
check_service "MLflow"              "http://localhost:5050"
check_service "Grafana"             "http://localhost:3000"
check_service "Prometheus"          "http://localhost:9090"


# Étape 5: Lancement de l’interface Airflow (optionnel)
read -p "🎈 Souhaites-tu démarrer Airflow ? (y/n): " run_airflow
if [[ "$run_airflow" == "y" ]]; then
    echo "🪁 Démarrage d’Airflow..."
    docker-compose --env-file .env.local -f docker-compose.airflow.yml up -d
    echo "🌬️  Interface Airflow → http://localhost:8080 (login: airflow / mdp: airflow)"
else
    echo "⏭️  Airflow non démarré."
fi

echo "🎉 Initialisation terminée !"
