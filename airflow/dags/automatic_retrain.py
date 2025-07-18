from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator
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
API_KEY = os.environ.get("API_KEY")

def wait_for_api():
    if not API_URL:
        raise AirflowSkipException("API_URL not set in environment")
    url = f"{API_URL}/health"
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
    headers = {"x-api-key": API_KEY}
    response = requests.post(f"{API_URL}/prepare_weekly_dataset", json={"current_week": current_week}, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Training failed: {response.text}")

def call_build_features():
    headers = {"x-api-key": API_KEY}
    response = requests.post(f"{API_URL}/build_features", headers=headers)
    if response.status_code != 200:
        raise Exception(f"Training failed: {response.text}")

def call_trainer_experiment_api():
    current_week = str(Variable.get("current_week", default_var=0))
    headers = {"x-api-key": API_KEY}
    response = requests.post(f"{API_URL}/trainer_experiment", json={"experiment_name": f"week_{current_week}"}, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Training failed: {response.text}")

def call_run_champion_selector_api():
    headers = {"x-api-key": API_KEY}
    response = requests.post(f"{API_URL}/run_champion_selector", headers=headers)
    if response.status_code != 200:
        raise Exception(f"Promotion failed: {response.text}")

def increment_week():
    week = int(Variable.get("current_week", default_var=0))
    Variable.set("current_week", week + 1)

def call_evidently_report():
    current_week = int(Variable.get("current_week", default_var=0))
    headers = {"x-api-key": API_KEY}
    response = requests.post(f"{API_URL}/run_evidently_report", json={"current_week": current_week}, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Evidently report failed: {response.text}")

def should_trigger_again():
    current_week = int(Variable.get("current_week", default_var=0))
    if current_week < 29:
        return "trigger_self"
    else:
        return "end_dag"

with DAG(
    dag_id="automatic_retrain",
    default_args=default_args,
    description="Auto retrain if new model is better",
    schedule_interval=None,  # No schedule, self-triggering
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

    evidently_report_task = PythonOperator(
        task_id='run_evidently_report',
        python_callable=call_evidently_report,
    )

    check_week = BranchPythonOperator(
        task_id="check_week",
        python_callable=should_trigger_again,
    )

    trigger_self = TriggerDagRunOperator(
        task_id="trigger_self",
        trigger_dag_id="automatic_retrain",
        wait_for_completion=False,
    )

    end_dag = EmptyOperator(
        task_id="end_dag"
    )

    # DAG structure
    api_available_task >> [prepare_weekly_dataset, evidently_report_task]
    prepare_weekly_dataset >> build_features >> trainer_experiment_task >> compare_and_promote_task >> increment_week_task
    increment_week_task >> check_week
    check_week >> [trigger_self, end_dag]