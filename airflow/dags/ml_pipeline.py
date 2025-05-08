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
            Mount(source='/home/ubuntu/MLOps_movie_recommandation/data', target='/app/data', type='bind'),
            Mount(source='/home/ubuntu/MLOps_movie_recommandation/models', target='/app/models', type='bind'),
            Mount(source='/home/ubuntu/MLOps_movie_recommandation/metrics', target='/app/metrics', type='bind'),
            Mount(source='/home/ubuntu/MLOps_movie_recommandation/airflow/dags', target='/opt/airflow/dags', type='bind'),
            Mount(source='/home/ubuntu/MLOps_movie_recommandation/airflow/logs', target='/opt/airflow/logs', type='bind'),
            Mount(source='/home/ubuntu/MLOps_movie_recommandation/airflow/plugins', target='/opt/airflow/plugins', type='bind'),
            Mount(source='/var/run/docker.sock', target='/var/run/docker.sock', type='bind'),
        ],
        force_pull=False,
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
        image='mlops_movie_recommandation_import_raw_data:latest',
        command='python import_raw_data.py',
    )

    build_features = create_docker_task(
        task_id='build_features',
        image='mlops_movie_recommandation_build-features:latest',
        command='python build_features.py',
    )

    train_model = create_docker_task(
        task_id='train_model',
        image='mlops_movie_recommandation_train:latest',
        command='python train.py',
    )

    predict_model = create_docker_task(
        task_id='predict_model',
        image='mlops_movie_recommandation_predict:latest',
        command='python predict.py',
    )

    evaluate_model = create_docker_task(
        task_id='evaluate_model',
        image='mlops_movie_recommandation_evaluate:latest',
        command='python evaluate.py',
    )

    # Définition des dépendances
    import_data >> build_features >> train_model >> predict_model >> evaluate_model