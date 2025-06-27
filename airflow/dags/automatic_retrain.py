from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from datetime import datetime
import requests
import json
import os
import subprocess
import yaml

API_URL = "http://localhost:8000"

#default_args = {
#    'start_date': datetime(2024, 1, 1),
#    'catchup': False,
#}

HOST_PATH = os.getenv("HOST_PATH")
#HOST_PATH = "/home/ubuntu/MLOps_movie_recommandation"

vol1 = os.path.join(HOST_PATH, 'data')
vol2 = os.path.join(HOST_PATH, 'models')
vol3 = os.path.join(HOST_PATH, 'metrics')
vol4 = os.path.join(HOST_PATH, 'airflow/dags')
vol5 = os.path.join(HOST_PATH, 'airflow/logs')
vol6 = os.path.join(HOST_PATH, 'airflow/plugins')

# Fonction pour créer un DockerOperator
def create_docker_task(task_id, image, command):
    return DockerOperator(
        task_id=task_id,
        image=image,
        command=command,
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        mounts=[
            Mount(source=vol1, target='/app/data', type='bind'),
            Mount(source=vol2, target='/app/models', type='bind'),
            Mount(source=vol3, target='/app/metrics', type='bind'),
            Mount(source=vol4, target='/opt/airflow/dags', type='bind'),
            Mount(source=vol5, target='/opt/airflow/logs', type='bind'),
            Mount(source=vol6, target='/opt/airflow/plugins', type='bind'),
            Mount(source='/var/run/docker.sock', target='/var/run/docker.sock', type='bind'),
        ],
        force_pull=False,
    )

def check_and_update_model():
    print("Comparing new model to current production...")
    with open("metrics/scores_model_prod.json") as f:
        current_metrics = json.load(f)

    # Cherche le meilleur run dans mlruns
    mlruns_dir = "mlruns"
    best_coverage = current_metrics.get("coverage_10", 0)
    best_model_path = None

    for root, dirs, files in os.walk(mlruns_dir):
        for file in files:
            if file == "metrics.json":
                with open(os.path.join(root, file)) as mf:
                    try:
                        m = json.load(mf)
                        if m.get("coverage_10", 0) > best_coverage:
                            best_coverage = m["coverage_10"]
                            best_model_path = os.path.join(root, "..", "artifacts", "model", "model.pkl")
                    except Exception as e:
                        continue

    if best_model_path:
        print(f"New model is better. Updating prod model. coverage_10: {best_coverage}")
        subprocess.run(["cp", best_model_path, "models/model.pkl"], check=True)
        subprocess.run(["dvc", "add", "models/model.pkl"], check=True)
        subprocess.run(["git", "commit", "-am", "Auto-update best model"], check=True)
        subprocess.run(["dvc", "push"], check=True)

        new_metrics_path = os.path.join(os.path.dirname(best_model_path), "..", "metrics.json")
        subprocess.run(["cp", new_metrics_path, "metrics/scores.json"], check=True)
        subprocess.run(["git", "add", "metrics/scores.json"], check=True)
        subprocess.run(["git", "commit", "-m", "Update metrics from best run"], check=True)
        subprocess.run(["git", "push"], check=True)
    else:
        print("No better model found. Keeping current production model.")


with DAG(
    dag_id="automatic_retrain",
    #default_args=default_args,
    description="Auto retrain if new model is better",
    schedule_interval="@weekly",  # Exécution manuelle
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'retrain'],
) as dag:
    
    experiment_task = create_docker_task(
        task_id='run_grid_search_task',
        image='experiment:latest',
        command='python src/experiment/entrypoint.sh',
    )

    check_update_task = PythonOperator(
        task_id="run_check_update_task",
        python_callable=check_and_update_model,
    )

    experiment_task >> check_update_task