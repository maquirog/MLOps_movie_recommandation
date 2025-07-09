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
from pathlib import Path

default_args = {
    'owner': 'airflow',
    'retries': 1,
}

HOST_PATH = os.getenv("HOST_PATH")

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
            "HOST_PROJECT_PATH": HOST_PATH
        },
        mounts=[
            Mount(source=os.path.join(HOST_PATH, 'src'), target='/app/src', type='bind'),
            Mount(source=os.path.join(HOST_PATH, 'mlruns'), target='/app/mlruns', type='bind'),
            Mount(source=os.path.join(HOST_PATH, 'models'), target='/app/models', type='bind'),
            Mount(source=os.path.join(HOST_PATH, 'metrics'), target='/app/metrics', type='bind'),
            Mount(source=os.path.join(HOST_PATH, '.git'), target='/app/.git', type='bind'),
            Mount(source=os.path.join(HOST_PATH, '.dvc'), target='/app/.dvc', type='bind'),
            Mount(source=os.path.join(HOST_PATH, '.env'), target='/app/.env', type='bind'),
            Mount(source=os.path.join(HOST_PATH, 'airflow/dags'), target='/opt/airflow/dags', type='bind'),
            Mount(source=os.path.join(HOST_PATH, 'airflow/logs'), target='/opt/airflow/logs', type='bind'),
            Mount(source=os.path.join(HOST_PATH, 'airflow/plugins'), target='/opt/airflow/plugins', type='bind'),
            Mount(source='/var/run/docker.sock', target='/var/run/docker.sock', type='bind'),
        ],
        force_pull=False,
    )

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
    #    command="bash -c 'nohup uvicorn src.api.main:app --host 0.0.0.0 --port 8000 > /dev/null 2>&1 &'",
    #    network_mode="mlops-net"
    #)
    
    experiment_task = create_docker_task(
        task_id='run_grid_search_task',
        image='experiment:latest',
        command="bash /app/src/experiment/entrypoint.sh",
        network_mode="mlops-net"
    )

    check_update_task = create_docker_task(
        task_id='run_check_update_prod_model_task',
        image='check_update_prod_model:latest',
        command="python /app/src/models/check_update_prod_model.py",
    )

    #api_task >> experiment_task >> check_update_task
    experiment_task >> check_update_task