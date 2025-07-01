from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
# from src.models.check_and_update_model import check_and_update_model
from datetime import datetime
import requests
import json
import os
import subprocess
import yaml
from pathlib import Path

default_args = {
    'owner': 'airflow',
    'retries': 1,
}

HOST_PATH = os.getenv("HOST_PATH")

vol1 = os.path.join(HOST_PATH, 'mlruns')
vol2 = os.path.join(HOST_PATH, 'models')
vol3 = os.path.join(HOST_PATH, 'metrics')
vol4 = os.path.join(HOST_PATH, 'airflow/dags')
vol5 = os.path.join(HOST_PATH, 'airflow/logs')
vol6 = os.path.join(HOST_PATH, 'airflow/plugins')

# Subclass DockerOperator to avoid command templating
class NoTemplateDockerOperator(DockerOperator):
    template_fields = tuple(f for f in DockerOperator.template_fields if f != "command")

# Fonction pour créer un DockerOperator
def create_docker_task(task_id, image, command, network_mode='bridge'):
    return NoTemplateDockerOperator(
        task_id=task_id,
        image=image,
        command=command,
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
        network_mode=network_mode,
        working_dir='/app',
        environment={
            "PYTHONPATH": "/app",
            "HOST_PROJECT_PATH": HOST_PATH,
        },
        mounts=[
            Mount(source=vol1, target='/app/mlruns', type='bind'),
            Mount(source=vol2, target='/app/models', type='bind'),
            Mount(source=vol3, target='/app/metrics', type='bind'),
            Mount(source=os.path.join(HOST_PATH, '.git'), target='/app/.git', type='bind'),
            Mount(source=os.path.join(HOST_PATH, '.dvc'), target='/app/.dvc', type='bind'),
            Mount(source=vol4, target='/opt/airflow/dags', type='bind'),
            Mount(source=vol5, target='/opt/airflow/logs', type='bind'),
            Mount(source=vol6, target='/opt/airflow/plugins', type='bind'),
            Mount(source='/var/run/docker.sock', target='/var/run/docker.sock', type='bind'),
        ],
        force_pull=False,
    )

#def check_and_update_model():
#    print("Comparing new model to current production...")
#    with open("metrics/scores_model_prod.json") as f:
#        current_metrics = json.load(f)
#
#    # Cherche le meilleur run dans mlruns
#    mlruns_dir = Path("mlruns").resolve()
#    best_coverage = current_metrics.get("coverage_10", 0)
#    best_model_path = None
#    best_metrics_path = None
#
#    for root, dirs, files in os.walk(mlruns_dir):
#        for file in files:
#            if file == "scores.json" and "artifacts/metrics" in str(root):
#                metrics_path = Path(root) / file
#                try:
#                    with open(metrics_path) as mf:
#                        m = json.load(mf)
#                        if m.get("coverage_10", 0) > best_coverage:
#                            run_dir = Path(root).parent.parent
#                            model_candidate = run_dir / "artifacts" / "model" / "model.pkl"
#                            if model_candidate.exists():
#                                best_coverage = m["coverage_10"]
#                                best_model_path = model_candidate
#                                best_metrics_path = metrics_path
#
#                except Exception as e:
#                    continue
#
#    if best_model_path and best_metrics_path:
#        print(f"✅ New model is better. Updating prod model. coverage_10: {best_coverage}")
#
#        # Copier new prod model + new prod model metrics
#        subprocess.run(["cp", best_model_path, "models/model_prod.pkl"], check=True)
#        subprocess.run(["cp", best_metrics_path, "metrics/scores_model_prod.json"], check=True)
#
#        # DVC commit prod model
#        subprocess.run(["dvc", "commit", "models/model_prod.pkl"], check=True)
#
#        # Git add + commit new prod model metrics
#        subprocess.run(["git", "add", "metrics/scores_model_prod.json"], check=True)
#        subprocess.run(["git", "commit", "-m", "Auto-update model_prod and scores_model_prod"], check=True)
#
#        # Git & DVC push
#        subprocess.run(["dvc", "push"], check=True)
#        subprocess.run(["git", "push"], check=True)
#    else:
#        print("⚠️ No better model found. Keeping current production model.")


with DAG(
    dag_id="automatic_retrain",
    default_args=default_args,
    description="Auto retrain if new model is better",
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'retrain'],
) as dag:

    #api_task = create_docker_task(
    #    task_id='launch_api_task',
    #    image='maquirog/api:latest',
    #    command="uvicorn src.api.main:app --host 0.0.0.0 --port 8000",
    #)

    experiment_task = create_docker_task(
        task_id='run_grid_search_task',
        image='experiment:latest',
        command="bash /app/src/experiment/entrypoint.sh",
        network_mode="mlops-net"
    )

    # check_update_task = PythonOperator(
    #     task_id="run_check_update_prod_model_task",
    #     python_callable=check_and_update_model,
    # )

    check_update_task = create_docker_task(
        task_id='run_check_update_prod_model_task',
        image='check_update_prod_model:latest',
        command="python /app/src/models/check_update_prod_model.py",
    )

    experiment_task >> check_update_task