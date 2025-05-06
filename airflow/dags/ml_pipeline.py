from airflow import DAG
from airflow.operators.docker_operator import DockerOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'retries': 1,
}

with DAG(
    'ml_pipeline',
    default_args=default_args,
    description='Pipeline ML orchestration with Docker',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:
    # Tâche pour importer les données brutes
    import_data = DockerOperator(
        task_id='import_data',
        image='data-service',
        command='python import_raw_data.py',
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
    )

    # Tâche pour construire les features
    build_features = DockerOperator(
        task_id='build_features',
        image='build-features',
        command='python build_features.py',
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
    )

    # Tâche pour entraîner le modèle
    train_model = DockerOperator(
        task_id='train_model',
        image='train',
        command='python train.py',
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
    )

    # Tâche pour évaluer le modèle
    evaluate_model = DockerOperator(
        task_id='evaluate_model',
        image='evaluate',
        command='python evaluate.py',
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
    )

    # Définition des dépendances
    import_data >> build_features >> train_model >> evaluate_model