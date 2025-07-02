from mlflow import MlflowClient
import mlflow
import requests
import os
import json
from itertools import product
import yaml
import datetime
import subprocess
import sys
import argparse
import time

API_URL = "http://api:8000"
#API_URL = "http://localhost:8000"
client = MlflowClient()
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# def create_new_experiment():
#     now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     experiment_name = f"weekly_experiment_{now}"
#     try:
#         experiment_id = client.create_experiment(experiment_name)
#     except:
#         experiment = client.get_experiment_by_name(experiment_name)
#         experiment_id = experiment.experiment_id
        
#     print(f"Created experiment: {experiment_name} ({experiment_id})")
#     return experiment_id

# import time

def get_experiment_id_by_name(name):
    for _ in range(10):
        experiment = client.get_experiment_by_name(name)
        if experiment:
            return experiment.experiment_id
        print(f"⏳ Waiting for experiment '{name}' to be registered...", flush=True)
        time.sleep(1)
    raise RuntimeError(f"Experiment '{name}' not found.")


def load_hyperparams_grid():
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hyperparameters.yaml')
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    hyperparam_grid = config.get("hyperparameters", {})
    keys = list(hyperparam_grid.keys())
    return list(product(*hyperparam_grid.values())), keys

# ### === Call API === ###
# def call_train(hyperparams, run_id=None):
#     payload = {"hyperparams": hyperparams}
#     if run_id:
#         payload["run_id"] = run_id
#     response = requests.post(f"{API_URL}/train", json=payload)
    
#     try:
#         response.raise_for_status()
#         print("✅ Train API called successfully.")
#         return response.json()  # Retourne la réponse JSON du serveur
#     except requests.HTTPError as e:
#         print(f"❌ Error calling Train API: {e}")
#         print("Response content:", response.text)
#         return None



# def call_predict(user_ids=None, n_recommendations=None):
#     json_data = {}
#     if user_ids:
#         json_data["user_ids"] = user_ids
#     if n_recommendations:
#         json_data["n_recommendations"] = n_recommendations

#     if json_data:
#         response = requests.post(f"{API_URL}/predict", json=json_data)
#     else:
#         response = requests.post(f"{API_URL}/predict")
#     response.raise_for_status()
#     print("✅ Predictions generated and saved.")
#     return response.json()

# def call_evaluate():
#     response = requests.post(f"{API_URL}/evaluate")
#     response.raise_for_status()
#     print("✅ Evaluation complete.")
#     return response.json()

### === Call en local === ###
def call_train(hyperparams, run_id):
    json_params = json.dumps(hyperparams)
    command = f"python ../models/train.py --hyperparams_dict '{json_params}' --run_id {run_id}"
    subprocess.run(command, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)

def call_predict(run_id=None):
    print("🧠 Predicting locally...", flush=True)
    command = "python ../models/predict.py"
    if run_id:
        command += f" --model_source runs:/{run_id}/model"
    subprocess.run(command, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)

def call_evaluate(run_id=None):
    print("📊 Evaluating locally...", flush=True)
    command = "python ../models/evaluate.py"
    if run_id:
        command += f" --run_id {run_id}"
    subprocess.run(command, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)


def train_evaluate_log_run(hyperparams, experiment_id):
    active_run = mlflow.active_run()
    if active_run and active_run.info.experiment_id == experiment_id:
        print("✅ Run active est dans la bonne expérience ✅", flush=True)
        # On réutilise la run active uniquement si elle est bien dans la bonne expérience
        run = active_run
        run_id = run.info.run_id
    else:
        print("🚨 Probleme la run n'est pas dans la bonne experience ‼️", flush=True)
        run = mlflow.start_run(experiment_id=experiment_id)
        run_id = run.info.run_id

    print(f"🔮🔮 grid search expID: {experiment_id} & run ID:{run_id}🔮🔮", flush=True)
    
    call_train(hyperparams, run_id)
    call_predict(run_id=run_id)
    call_evaluate(run_id=run_id)

    metrics_path = os.path.join(BASE_DIR, "metrics", "scores.json")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # mlflow.log_metrics(metrics)
    # mlflow.log_params(hyperparams)

    if not (active_run and active_run.info.experiment_id == experiment_id):
        mlflow.end_run()

    print(f"🔁 Run {run_id} logged with metrics: {metrics}", flush=True)
    return run_id, metrics

# def save_best_run_info(experiment_id, best_run_id):
#     shared_dir = os.path.join(BASE_DIR, "shared")
#     os.makedirs(shared_dir, exist_ok=True)
#     path = os.path.join(shared_dir, "best_run.json")
#     with open(path, "w") as f:
#         json.dump({"experiment_id": experiment_id, "best_run_id": best_run_id}, f)

def register_model(run_id, model_name="movie_recommender"):
    client = MlflowClient()
    model_version = client.create_model_version(
        name=model_name,
        source=f"runs:/{run_id}/model",
        run_id=run_id
    )
    print(f"Model version {model_version.version} registered under '{model_name}'", flush=True)
    return model_version.version

def set_model_alias(model_name, version, alias="Challenger"):
    client = MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        version=version,
        alias=alias)
    print(f"[Registry] Model version {version} now aliased as '{alias}'")
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    experiment_name = args.experiment_name
    
    # Set l'expérience AVANT de récupérer l'ID ou de commencer les runs
    mlflow.set_tracking_uri("http://127.0.0.1:5050")
    
    experiment_id = get_experiment_id_by_name(experiment_name)
    print(f"Experiment id créée: {experiment_name} ({experiment_id})", flush=True)  
    # experiment_id = create_new_experiment()

    param_combinations, keys = load_hyperparams_grid()

    
    best = {"coverage": -1, "run_id": None, "params": None}


    for values in param_combinations:
        hyperparams = dict(zip(keys, values))
        print(f"🏋️‍♂️ Training with hyperparameters: {hyperparams}")
        run_id, metrics = train_evaluate_log_run(hyperparams, experiment_id)
        if metrics.get("coverage_10", 0) > best["coverage"]:
                best.update({
                    "coverage": metrics["coverage_10"],
                    "run_id": run_id,
                    "params": hyperparams
                })


    print("\n=== 🏆 Best Run Summary ===")
    print(f"Run ID       : {best['run_id']}")
    print(f"Hyperparams  : {best['params']}")
    print(f"Coverage_10  : {best['coverage']}")
    
    
    # save_best_run_info(experiment_id, best["run_id"])
    version = register_model(best["run_id"])
    set_model_alias("movie_recommender", version, "Challenger")

if __name__ == "__main__":
    main()