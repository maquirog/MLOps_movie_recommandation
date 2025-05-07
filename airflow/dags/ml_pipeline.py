from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'retries': 1,
}

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
            Mount(source='/opt/airflow/dags', target='/opt/airflow/dags', type='bind'),
            Mount(source='/opt/airflow/logs', target='/opt/airflow/logs', type='bind'),
            Mount(source='/opt/airflow/plugins', target='/opt/airflow/plugins', type='bind'),
            Mount(source='/var/run/docker.sock', target='/var/run/docker.sock', type='bind'),
        ],
    )

# Définition du DAG
with DAG(
    'ml_pipeline',
    default_args=default_args,
    description='Pipeline ML orchestration with Docker',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:
    # Création des tâches
    import_data = create_docker_task(
        task_id='import_data',
        image='import_raw_data',
        command='python import_raw_data.py',
    )

    build_features = create_docker_task(
        task_id='build_features',
        image='build-features',
        command='python build_features.py',
    )

    train_model = create_docker_task(
        task_id='train_model',
        image='train',
        command='python train.py',
    )

    evaluate_model = create_docker_task(
        task_id='evaluate_model',
        image='evaluate',
        command='python evaluate.py',
    )

    # Définition des dépendances
    import_data >> build_features >> train_model >> evaluate_model