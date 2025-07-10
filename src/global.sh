# mettre à jour .env.local
# créer réseau ml-net
docker network create mlops-net
# importer les data et créer les feature
docker-compose up import_raw_data
docker-compose up build_features
# lancer api
docker-compose up api
# lancer airflow
docker-compose --env-file .env.local -f docker-compose.airflow.yml up