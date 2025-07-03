from mlflow import MlflowClient
import mlflow
import requests
import os
import json
from itertools import product
import yaml
import subprocess
import sys
import argparse
import time
from mlflow.exceptions import MlflowException

# === 🌍 Variables d'environnement === #
API_URL = os.environ.get("API_URL", "http://api:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "movie_recommender")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5050")
BASE_DIR = os.environ.get("BASE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
DATA_DIR= os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "models"))
METRIC_KEY = os.environ.get("METRIC_KEY", "ndcg_10")

METRICS_DIR = os.environ.get("METRICS_DIR", os.path.join(BASE_DIR, "metrics"))
PREDICT_DIR = os.path.join(DATA_DIR, "predictions")

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

## === Call API === ###
def call_train(hyperparams, run_id=None):
    payload = {"hyperparams": hyperparams}
    if run_id:
        payload["run_id"] = run_id
    response = requests.post(f"{API_URL}/train", 
                             json=payload,
                             headers={"Content-Type": "application/json"},
                             timeout=300
                             )
    
    try:
        response.raise_for_status()
        print("✅ Train API called successfully.", flush=True)
        return response.json()  # Retourne la réponse JSON du serveur
    except requests.HTTPError as e:
        print(f"❌ Error calling Train API: {e}", flush=True)
        print("Response content:", response.text, flush=True)
        return None


def call_predict(model_source=None, output_filename=None):
    payload = {
        "model_source":model_source,
        "output_filename": output_filename
    }

    try:
        print(f"📡 Envoi requête à {API_URL}/predict ...", flush=True)
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300
        )
        response.raise_for_status()  # Raise HTTPError for bad status codes
        data = response.json()
        print("✅ Prédiction lancée avec succès via l'API.", flush=True)
        return data
    except requests.RequestException as e:
        print(f"❌ Erreur lors de l'appel à l'API /predict : {e}", flush=True)
        return None


def call_evaluate(run_id=None, input_filename=None, output_filename = None):
    payload = {"run_id": run_id}
    if input_filename:
        payload["input_filename"] = input_filename
    if output_filename:
        payload["output_filename"] = output_filename

    try:
        print(f"📡 Envoi requête à {API_URL}/evaluate ...", flush=True)
        response = requests.post(
            f"{API_URL}/evaluate",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300
        )
        response.raise_for_status()
        data = response.json()
        print("✅ Évaluation réussie via l’API.", flush=True)
        return data
    except requests.RequestException as e:
        print(f"❌ Erreur lors de l'appel à l'API /evaluate : {e}", flush=True)
        return None


### === Call en local === ###
# def call_train(hyperparams, run_id):
#     json_params = json.dumps(hyperparams)
#     command = f"python ../models/train.py --hyperparams_dict '{json_params}' --run_id {run_id}"
#     subprocess.run(command, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)

# def call_predict(run_id=None, output_filename = None):
#     print("🧠 Predicting locally...", flush=True)
#     command = "python ../models/predict.py"
#     if run_id:
#         command += f" --model_source runs:/{run_id}/model"
#     if output_filename:
#         command += f" --output_filename {output_filename}"
#     subprocess.run(command, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)

# def call_evaluate(run_id=None, input_filename = None):
#     print("📊 Evaluating locally...", flush=True)
#     command = "python ../models/evaluate.py"
#     if run_id:
#         command += f" --run_id {run_id}"
        
#     if input_filename:
#         command += f" --input_filename {input_filename}"

#     try:
#         result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
#     except subprocess.CalledProcessError as e:
#         print("⚠️ Evaluate script crashed.")
#         print("stdout:", e.stdout)
#         print("stderr:", e.stderr)
#         return None
    
#     first_line = result.stdout.strip().splitlines()[0]

#     try:
#         metrics = json.loads(first_line)
#         print("✅ Metrics récupérées :", metrics)
#         return metrics
#     except json.JSONDecodeError:
#         print("⚠️ Impossible de parser les métriques retournées.")
#         print("Sortie brute :", result.stdout)
#         return None
    
# === MLFlow Run === #

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
    
    # Paths
    model_source = os.path.join(MODELS_DIR, f"model_{run_id}.pkl")
    predictions_filename = f"predictions_{run_id}.json"
    metrics_filename = os.path.join(METRICS_DIR, f"scores_{run_id}.json")
    
    # === Pipeline calls === #
    call_train(hyperparams, run_id)
    call_predict(model_source, output_filename = predictions_filename)
    call_evaluate(run_id=run_id, input_filename = predictions_filename, output_filename = metrics_filename)
    
    with open(metrics_filename, "r") as f:
        metrics = json.load(f)
            
    # === Clean exit === #
    if not (active_run and active_run.info.experiment_id == experiment_id):
        mlflow.end_run()

    print(f"📈 Run {run_id} terminée avec succès. Metrics : {metrics}", flush=True)
    return run_id, metrics

# === Post-processing === #

def clean_dirs(best_run_id):
    # === Nettoyage prédictions ===
    for f in os.listdir(PREDICT_DIR):
        full_path = os.path.join(PREDICT_DIR, f)
        if f != f"predictions_{best_run_id}.json":
            os.remove(full_path)

    pred_old = os.path.join(PREDICT_DIR, f"predictions_{best_run_id}.json")
    pred_new = os.path.join(PREDICT_DIR, "predictions_challenger.json")
    if os.path.exists(pred_old):
        os.rename(pred_old, pred_new)
    else:
        print(f"⚠️ Fichier {pred_old} introuvable pour renommage.", flush=True)

    # === Nettoyage metrics ===
    if METRICS_DIR:
        for f in os.listdir(METRICS_DIR):
            full_path = os.path.join(METRICS_DIR, f)
            if f != f"scores_{best_run_id}.json":
                os.remove(full_path)

        metric_old = os.path.join(METRICS_DIR, f"scores_{best_run_id}.json")
        metric_new = os.path.join(METRICS_DIR, "challenger_scores.json")
        if os.path.exists(metric_old):
            os.rename(metric_old, metric_new)
        else:
            print(f"⚠️ Fichier {metric_old} introuvable pour renommage.", flush=True)

    # === Nettoyage models ===
    for f in os.listdir(MODELS_DIR):
        full_path = os.path.join(MODELS_DIR, f)
        if f not in [f"model_{best_run_id}.pkl", "model_champion.pkl"]:
            os.remove(full_path)

    model_old = os.path.join(MODELS_DIR, f"model_{best_run_id}.pkl")
    model_new = os.path.join(MODELS_DIR, "model_challenger.pkl")
    if os.path.exists(model_old):
        os.rename(model_old, model_new)
    else:
        print(f"⚠️ Fichier {model_old} introuvable pour renommage.", flush=True)

def register_challenger(run_id, alias="challenger", model_name=MODEL_NAME, metrics_dir = METRICS_DIR):
    print("💾 New Challenger defined, time to register ", flush=True)
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

def set_model_alias(version, alias="challenger", model_name=MODEL_NAME):
    client = MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        version=version,
        alias=alias)
    print(f"[Registry] Model version {version} now aliased as '{alias}'", flush=True)

# === Entrypoint === #

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    experiment_name = args.experiment_name
    
    # Set l'expérience AVANT de récupérer l'ID ou de commencer les runs
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    experiment_id = get_experiment_id_by_name(experiment_name)
    print(f"Experiment id créée: {experiment_name} ({experiment_id})", flush=True)  


    param_combinations, keys = load_hyperparams_grid()

    
    best = {METRIC_KEY: -1, "run_id": None, "params": None}


    for values in param_combinations:
        hyperparams = dict(zip(keys, values))
        print(f"🏋️‍♂️ Training with hyperparameters: {hyperparams}", flush=True)
        run_id, metrics = train_predict_evaluate_log_run(hyperparams, experiment_id)
        if metrics.get(METRIC_KEY, 0) > best[METRIC_KEY]:
                best.update({
                    METRIC_KEY: metrics[METRIC_KEY],
                    "run_id": run_id,
                    "params": hyperparams
                })
                


    print("\n=== 🏆 Best Run Summary ===", flush=True)
    print(f"Run ID       : {best['run_id']}", flush=True)
    print(f"Hyperparams  : {best['params']}", flush=True)
    print(f"{METRIC_KEY}  : {best[METRIC_KEY]}", flush=True)
    
    
    clean_dirs(best["run_id"])
    version = register_challenger(best["run_id"])
    set_model_alias(version, "challenger")

if __name__ == "__main__":
    main()