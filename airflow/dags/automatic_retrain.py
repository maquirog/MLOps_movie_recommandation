from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException
from airflow.models import Variable
import os
import requests
import time
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'depends_on_past': False,
}
API_URL = os.environ.get("API_URL")

current_week = int(Variable.get("current_week", default_var=0))

def wait_for_api():
    if not API_URL:
        raise AirflowSkipException("API_URL not set in environment")
    
    url = f"{API_URL}/health"  # ou un endpoint simple qui répond vite
    max_retries = 10
    wait_seconds = 5
    for i in range(max_retries):
        try:
            r = requests.get(url)
            if r.status_code == 200:
                print("API is available!")
                return
        except Exception:
            pass
        print(f"API not available yet, retrying {i+1}/{max_retries}...")
        time.sleep(wait_seconds)
    raise AirflowSkipException("API is not available after retries")


def call_prepare_weekly_dataset():
    current_week = int(Variable.get("current_week", default_var=0))
    response = requests.post(f"{API_URL}/prepare_weekly_dataset", json={"current_week": current_week})
    if response.status_code != 200:
        raise Exception(f"Training failed: {response.text}")

def call_build_features():
    response = requests.post(f"{API_URL}/build_features")
    if response.status_code != 200:
        raise Exception(f"Training failed: {response.text}")

def call_trainer_experiment_api():
    current_week = str(Variable.get("current_week", default_var=0))
    response = requests.post(f"{API_URL}/trainer_experiment", json={"experiment_name": "string",})
    if response.status_code != 200:
        raise Exception(f"Training failed: {response.text}")

def call_run_champion_selector_api():
    response = requests.post(f"{API_URL}/run_champion_selector")
    if response.status_code != 200:
        raise Exception(f"Promotion failed: {response.text}")

def increment_week():
    week = int(Variable.get("current_week", default_var=0))
    Variable.set("current_week", week + 1)

with DAG(
    dag_id="automatic_retrain",
    default_args=default_args,
    description="Auto retrain if new model is better",
    schedule_interval=None,   #@weekly
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'retrain'],
    max_active_runs=1,
) as dag:

    api_available_task = PythonOperator(
        task_id='wait_for_api',
        python_callable=wait_for_api
    )
    
    prepare_weekly_dataset = PythonOperator(
        task_id='prepare_weekly_dataset',
        python_callable=call_prepare_weekly_dataset,
    )
    
    build_features = PythonOperator(
        task_id='build_features',
        python_callable=call_build_features,
    )
    
    trainer_experiment_task = PythonOperator(
        task_id='launch_weekly_experiment_mlflow',
        python_callable=call_trainer_experiment_api,
    )

    compare_and_promote_task = PythonOperator(
        task_id='compare_and_promote_task',
        python_callable=call_run_champion_selector_api,
    )
    
    increment_week_task = PythonOperator(
        task_id='increment_week',
        python_callable=increment_week,
    )

    # api_available_task >> trainer_experiment_task >> compare_and_promote_task >> increment_week_task
    api_available_task >> prepare_weekly_dataset >> build_features >> trainer_experiment_task >> compare_and_promote_task >> increment_week_task