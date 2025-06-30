from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from docker.types import Mount
from datetime import datetime
import os

default_args = {
    'owner': 'airflow',
    'retries': 1,
}

HOST_PATH = os.getenv("HOST_PATH")

vol1 = os.path.join(HOST_PATH, 'data')
vol2 = os.path.join(HOST_PATH, 'models')
vol3 = os.path.join(HOST_PATH, 'metrics')
vol4 = os.path.join(HOST_PATH, 'airflow/dags')
vol5 = os.path.join(HOST_PATH, 'airflow/logs')
vol6 = os.path.join(HOST_PATH, 'airflow/plugins')

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

def get_current_week():
    # Récupère la semaine courante depuis la variable Airflow, default=0
    return int(Variable.get("current_week", default_var=0))

def increment_week():
    week = int(Variable.get("current_week", default_var=0))
    Variable.set("current_week", week + 1)

with DAG(
    'ml_pipeline',
    default_args=default_args,
    description='Pipeline ML orchestration with Docker',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    import_data = create_docker_task(
        task_id='import_data',
        image='maquirog/import_raw_data:latest',
        command='python src/data/import_raw_data.py',
    )

    current_week = get_current_week()
    prepare_weekly_dataset = create_docker_task(
        task_id='prepare_weekly_dataset',
        image='maquirog/prepare_weekly_dataset:latest',
        command=f'python src/data/prepare_weekly_dataset.py {current_week}',
    )

    build_features = create_docker_task(
        task_id='build_features',
        image='maquirog/build_features:latest',
        command='python src/data/build_features.py',
    )

    train_model = create_docker_task(
        task_id='train_model',
        image='maquirog/train:latest',
        command='python src/models/train.py',
    )

    predict_model = create_docker_task(
        task_id='predict_model',
        image='maquirog/predict:latest',
        command='python src/models/predict.py',
    )

    evaluate_model = create_docker_task(
        task_id='evaluate_model',
        image='maquirog/evaluate:latest',
        command='python src/models/evaluate.py',
    )

    increment_week_task = PythonOperator(
        task_id='increment_week',
        python_callable=increment_week,
    )

    # Dépendances du pipeline
    import_data >> prepare_weekly_dataset >> build_features >> train_model >> predict_model >> evaluate_model >> increment_week_task