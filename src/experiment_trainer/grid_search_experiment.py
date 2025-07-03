from mlflow import MlflowClient
import mlflow
import requests
import os
import json
from itertools import product
import yaml
# import datetime
import subprocess
import sys
import argparse
import time
from mlflow.exceptions import MlflowException

# === 🌍 Variables d'environnement === #
API_URL = os.environ.get("API_URL", "http://localhost:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "movie_recommender")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5050")
BASE_DIR = os.environ.get("BASE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
DATA_DIR= os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
METRIC_KEY = os.environ.get("METRIC_KEY", "ndcg_10")

# METRICS_DIR = os.environ.get("METRICS_DIR", os.path.join(BASE_DIR, "metrics"))
METRICS_DIR = None
PREDICT_DIR = os.path.join(DATA_DIR, "predictions")

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

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

def call_predict(run_id=None, output_filename = None):
    print("🧠 Predicting locally...", flush=True)
    command = "python ../models/predict.py"
    if run_id:
        command += f" --model_source runs:/{run_id}/model"
    if output_filename:
        command += f" --output_filename {output_filename}"
    subprocess.run(command, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)

def call_evaluate(run_id=None, input_filename = None):
    print("📊 Evaluating locally...", flush=True)
    command = "python ../models/evaluate.py"
    if run_id:
        command += f" --run_id {run_id}"
        
    if input_filename:
        command += f" --input_filename {input_filename}"

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print("⚠️ Evaluate script crashed.")
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)
        return None
    
    first_line = result.stdout.strip().splitlines()[0]

    try:
        metrics = json.loads(first_line)
        print("✅ Metrics récupérées :", metrics)
        return metrics
    except json.JSONDecodeError:
        print("⚠️ Impossible de parser les métriques retournées.")
        print("Sortie brute :", result.stdout)
        return None
    
def train_predict_evaluate_log_run(hyperparams, experiment_id):
    active_run = mlflow.active_run()
    if active_run and active_run.info.experiment_id == experiment_id:
        print("✅ Run active est dans la bonne expérience ✅", flush=True)
        # On réutilise la run active uniquement si elle est bien dans la bonne expérience
        run = active_run
        run_id = run.info.run_id
    else:
        # on previent erreur mais lance quand meme pour l'instant
        print("🚨 Probleme la run n'est pas dans la bonne experience ‼️", flush=True)
        run = mlflow.start_run(experiment_id=experiment_id)
        run_id = run.info.run_id

    print(f"🔮🔮 grid search expID: {experiment_id} & run ID:{run_id}🔮🔮", flush=True)
    
    call_train(hyperparams, run_id)
    
    predictions_filename = f"predictions_{run_id}.json"
    
    call_predict(run_id=run_id, output_filename = predictions_filename)
    metrics = call_evaluate(run_id=run_id, input_filename=predictions_filename) # ou mettre jour pour lire les metric depuis la run
    
    print("\n📊 Recommandation Evaluation Metrics")
    for key, val in metrics.items():
        print(f"{key}: {val}")
            
    if not (active_run and active_run.info.experiment_id == experiment_id):
        mlflow.end_run()

    print(f"🔁 Run {run_id} logged with metrics: {metrics}", flush=True)
    return run_id, metrics

def clean_dirs(run_id):  
    # Nettoyage prédictions
    files = os.listdir(PREDICT_DIR)
    # Supprime tout sauf celui à garder
    for f in files:
        if f != f"predictions_{run_id}.json":
            os.remove(os.path.join(PREDICT_DIR, f))
    # Renomme après suppression
    old_path = os.path.join(PREDICT_DIR, f"predictions_{run_id}.json")
    new_path = os.path.join(PREDICT_DIR, "predictions_challenger.json")
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
    else:
        print(f"⚠️ Fichier {old_path} introuvable pour renommage.")
            
    if METRICS_DIR:
        files = os.listdir(METRICS_DIR)
        for f in files:
            if f != f"{run_id}.json":
                os.remove(os.path.join(METRICS_DIR, f))
        old_path = os.path.join(METRICS_DIR, f"{run_id}.json")
        new_path = os.path.join(METRICS_DIR, "challenger_score.json")
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
        else:
            print(f"⚠️ Fichier {old_path} introuvable pour renommage.")

def register_challenger(run_id, alias="Challenger", model_name=MODEL_NAME, metrics_dir = METRICS_DIR):
    print("💾 New Challenger defined, time to register ")
    client = MlflowClient()
    
    try:
        client.get_registered_model(model_name)
    except MlflowException:
        client.create_registered_model(model_name)
    
    model_version = client.create_model_version(
        name=model_name,
        source=f"runs:/{run_id}/model",
        run_id=run_id
    )
    
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tags({"model_version": model_version.version, "alias": alias, "model_name": model_name})
        if metrics_dir:
            metrics_path = os.path.join(metrics_dir, "challenger_score.json")
            mlflow.log_artifact(metrics_path)
    print(f"✅ Model {model_name} version {model_version.version} enregistré et aliasé comme '{alias}'", flush=True)
    
    return model_version.version

def set_model_alias(version, alias="Challenger", model_name=MODEL_NAME):
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
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5050"))
    
    experiment_id = get_experiment_id_by_name(experiment_name)
    print(f"Experiment id créée: {experiment_name} ({experiment_id})", flush=True)  


    param_combinations, keys = load_hyperparams_grid()

    
    best = {METRIC_KEY: -1, "run_id": None, "params": None}


    for values in param_combinations:
        hyperparams = dict(zip(keys, values))
        print(f"🏋️‍♂️ Training with hyperparameters: {hyperparams}")
        run_id, metrics = train_predict_evaluate_log_run(hyperparams, experiment_id)
        if metrics.get(METRIC_KEY, 0) > best[METRIC_KEY]:
                best.update({
                    METRIC_KEY: metrics[METRIC_KEY],
                    "run_id": run_id,
                    "params": hyperparams
                })
                


    print("\n=== 🏆 Best Run Summary ===")
    print(f"Run ID       : {best['run_id']}")
    print(f"Hyperparams  : {best['params']}")
    print(f"{METRIC_KEY}  : {best[METRIC_KEY]}")
    
    
    clean_dirs(best["run_id"])
    version = register_challenger(best["run_id"])
    set_model_alias(version, "Challenger")

if __name__ == "__main__":
    main()